//! Lightweight frame pass graph with backward dead-pass pruning.
//!
//! Each logical stage in [`crate::ViewportHost`] declares the frame resources it reads and writes.
//! Before command encoding starts, [`PassGraph::compile`] walks backward from the active frame
//! sinks (`FINAL_OUTPUT` and optional `SELECTION_READBACK`) and prunes any unreferenced stage,
//! while deriving whether downstream passes require preserving scene depth or MSAA color buffers.

/// Bitmask of logical frame resources produced and consumed by viewport passes.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(crate) struct FrameResource(u32);

impl FrameResource {
    pub(crate) const NONE: Self = Self(0);
    pub(crate) const PRE_SCENE: Self = Self(1 << 0);
    pub(crate) const GPU_CULL_P1: Self = Self(1 << 1);
    pub(crate) const SCENE_COLOR: Self = Self(1 << 2);
    pub(crate) const SCENE_DEPTH: Self = Self(1 << 3);
    pub(crate) const SCENE_STENCIL: Self = Self(1 << 4);
    pub(crate) const HIZ_PYRAMID: Self = Self(1 << 5);
    pub(crate) const GPU_CULL_P2: Self = Self(1 << 6);
    pub(crate) const SELECTION_ID: Self = Self(1 << 7);
    pub(crate) const RESOLVED_COLOR: Self = Self(1 << 8);
    pub(crate) const OUTLINE_MASK: Self = Self(1 << 9);
    pub(crate) const OUTLINE_JFA: Self = Self(1 << 10);
    pub(crate) const FINAL_OUTPUT: Self = Self(1 << 11);
    pub(crate) const SELECTION_READBACK: Self = Self(1 << 12);

    #[inline]
    pub(crate) const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[inline]
    pub(crate) const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }
}

/// Ordered logical stages that may participate in a single viewport frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(usize)]
pub(crate) enum PassId {
    PreSceneHooks = 0,
    GpuCullPhase1 = 1,
    OutlineMask = 2,
    OutlineJfa = 3,
    ScenePhase1 = 4,
    HiZBuild = 5,
    GpuCullPhase2 = 6,
    ScenePhase2 = 7,
    PostOpaqueHooks = 8,
    SceneTransparent = 9,
    SceneResolve = 10,
    PostSceneResolvedHooks = 11,
    DynamicComposite = 12,
    SelectionReduce = 13,
}

const PASS_COUNT: usize = 14;

/// Inputs describing active features and layer requests for the current frame.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct FramePlanInput {
    pub(crate) rebuild_scene: bool,
    pub(crate) use_gpu_cull: bool,
    pub(crate) use_mrt_selection: bool,
    pub(crate) run_selection_reduce: bool,
    pub(crate) has_stencil: bool,
    pub(crate) has_pre_scene_hook: bool,
    pub(crate) has_post_opaque_hook: bool,
    pub(crate) has_transparent_layers: bool,
    pub(crate) has_post_scene_resolved_hook: bool,
    pub(crate) has_outline: bool,
    pub(crate) need_dynamic: bool,
    pub(crate) msaa_samples: u32,
    pub(crate) continuous_second_pass: bool,
}

#[derive(Debug, Clone, Copy)]
struct PassNode {
    id: PassId,
    enabled: bool,
    reads: FrameResource,
    writes: FrameResource,
}

/// Compiled frame execution plan produced with zero heap allocations.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PassGraph {
    live: [bool; PASS_COUNT],
    #[cfg(test)]
    enabled_count: u32,
    #[cfg(test)]
    live_count: u32,
    keep_depth_after_primary: bool,
    keep_depth_after_phase2: bool,
    keep_msaa_after_primary: bool,
    keep_msaa_after_phase2: bool,
    separate_transparent_pass: bool,
}

impl PassGraph {
    /// Builds and prunes the frame DAG for `input` using backward reachability from frame sinks.
    #[must_use]
    pub(crate) fn compile(input: FramePlanInput) -> Self {
        let separate_transparent_pass = input.rebuild_scene && input.has_post_opaque_hook;
        let scene_primary_writes = FrameResource::SCENE_COLOR
            .union(FrameResource::SCENE_DEPTH)
            .union(if input.has_stencil {
                FrameResource::SCENE_STENCIL
            } else {
                FrameResource::NONE
            })
            .union(if input.use_mrt_selection {
                FrameResource::SELECTION_ID
            } else {
                FrameResource::NONE
            })
            .union(if !input.use_gpu_cull && !separate_transparent_pass {
                FrameResource::RESOLVED_COLOR
            } else {
                FrameResource::NONE
            });

        let scene_phase2_writes = FrameResource::SCENE_COLOR
            .union(if input.use_mrt_selection {
                FrameResource::SELECTION_ID
            } else {
                FrameResource::NONE
            })
            .union(if !separate_transparent_pass {
                FrameResource::RESOLVED_COLOR
            } else {
                FrameResource::NONE
            });

        let needs_explicit_resolve =
            separate_transparent_pass && !input.has_transparent_layers && input.msaa_samples > 1;
        let transparent_writes = FrameResource::SCENE_COLOR.union(FrameResource::RESOLVED_COLOR);
        let post_opaque_writes = FrameResource::SCENE_COLOR
            .union(if input.has_stencil {
                FrameResource::SCENE_STENCIL
            } else {
                FrameResource::NONE
            })
            .union(
                if !input.has_transparent_layers && input.msaa_samples <= 1 {
                    FrameResource::RESOLVED_COLOR
                } else {
                    FrameResource::NONE
                },
            );

        let post_resolved_enabled = input.rebuild_scene && input.has_post_scene_resolved_hook;
        let dynamic_writes = FrameResource::FINAL_OUTPUT;
        let scene_is_final = !input.need_dynamic;

        let nodes = [
            PassNode {
                id: PassId::PreSceneHooks,
                enabled: input.rebuild_scene && input.has_pre_scene_hook,
                reads: FrameResource::NONE,
                writes: FrameResource::PRE_SCENE,
            },
            PassNode {
                id: PassId::GpuCullPhase1,
                enabled: input.rebuild_scene && input.use_gpu_cull,
                reads: FrameResource::NONE,
                writes: FrameResource::GPU_CULL_P1,
            },
            PassNode {
                id: PassId::OutlineMask,
                enabled: input.has_outline && input.need_dynamic,
                reads: FrameResource::NONE,
                writes: FrameResource::OUTLINE_MASK,
            },
            PassNode {
                id: PassId::OutlineJfa,
                enabled: input.has_outline && input.need_dynamic,
                reads: FrameResource::OUTLINE_MASK,
                writes: FrameResource::OUTLINE_JFA,
            },
            PassNode {
                id: PassId::ScenePhase1,
                enabled: input.rebuild_scene,
                reads: if input.has_pre_scene_hook {
                    FrameResource::PRE_SCENE
                } else {
                    FrameResource::NONE
                }
                .union(if input.use_gpu_cull {
                    FrameResource::GPU_CULL_P1
                } else {
                    FrameResource::NONE
                }),
                writes: scene_primary_writes.union(
                    if scene_is_final
                        && !input.use_gpu_cull
                        && !separate_transparent_pass
                        && !post_resolved_enabled
                    {
                        FrameResource::FINAL_OUTPUT
                    } else {
                        FrameResource::NONE
                    },
                ),
            },
            PassNode {
                id: PassId::HiZBuild,
                enabled: input.rebuild_scene && input.use_gpu_cull,
                reads: FrameResource::SCENE_DEPTH,
                writes: FrameResource::HIZ_PYRAMID,
            },
            PassNode {
                id: PassId::GpuCullPhase2,
                enabled: input.rebuild_scene && input.use_gpu_cull,
                reads: FrameResource::HIZ_PYRAMID,
                writes: FrameResource::GPU_CULL_P2,
            },
            PassNode {
                id: PassId::ScenePhase2,
                enabled: input.rebuild_scene && input.use_gpu_cull,
                reads: FrameResource::GPU_CULL_P2
                    .union(FrameResource::SCENE_COLOR)
                    .union(FrameResource::SCENE_DEPTH)
                    .union(if input.use_mrt_selection {
                        FrameResource::SELECTION_ID
                    } else {
                        FrameResource::NONE
                    }),
                writes: scene_phase2_writes.union(
                    if scene_is_final && !separate_transparent_pass && !post_resolved_enabled {
                        FrameResource::FINAL_OUTPUT
                    } else {
                        FrameResource::NONE
                    },
                ),
            },
            PassNode {
                id: PassId::PostOpaqueHooks,
                enabled: separate_transparent_pass,
                reads: FrameResource::SCENE_COLOR
                    .union(FrameResource::SCENE_DEPTH)
                    .union(if input.has_stencil {
                        FrameResource::SCENE_STENCIL
                    } else {
                        FrameResource::NONE
                    }),
                writes: post_opaque_writes.union(
                    if scene_is_final
                        && !input.has_transparent_layers
                        && !needs_explicit_resolve
                        && !post_resolved_enabled
                    {
                        FrameResource::FINAL_OUTPUT
                    } else {
                        FrameResource::NONE
                    },
                ),
            },
            PassNode {
                id: PassId::SceneTransparent,
                enabled: separate_transparent_pass && input.has_transparent_layers,
                reads: FrameResource::SCENE_COLOR.union(FrameResource::SCENE_DEPTH),
                writes: transparent_writes.union(if scene_is_final && !post_resolved_enabled {
                    FrameResource::FINAL_OUTPUT
                } else {
                    FrameResource::NONE
                }),
            },
            PassNode {
                id: PassId::SceneResolve,
                enabled: needs_explicit_resolve,
                reads: FrameResource::SCENE_COLOR,
                writes: FrameResource::RESOLVED_COLOR.union(
                    if scene_is_final && !post_resolved_enabled {
                        FrameResource::FINAL_OUTPUT
                    } else {
                        FrameResource::NONE
                    },
                ),
            },
            PassNode {
                id: PassId::PostSceneResolvedHooks,
                enabled: post_resolved_enabled,
                reads: FrameResource::RESOLVED_COLOR.union(FrameResource::SCENE_DEPTH),
                writes: FrameResource::RESOLVED_COLOR.union(if scene_is_final {
                    FrameResource::FINAL_OUTPUT
                } else {
                    FrameResource::NONE
                }),
            },
            PassNode {
                id: PassId::DynamicComposite,
                enabled: input.need_dynamic,
                reads: FrameResource::RESOLVED_COLOR.union(if input.has_outline {
                    FrameResource::OUTLINE_JFA
                } else {
                    FrameResource::NONE
                }),
                writes: dynamic_writes,
            },
            PassNode {
                id: PassId::SelectionReduce,
                enabled: input.run_selection_reduce,
                reads: FrameResource::SELECTION_ID,
                writes: FrameResource::SELECTION_READBACK,
            },
        ];

        let mut live = [false; PASS_COUNT];
        #[cfg(test)]
        let mut enabled_count = 0u32;
        #[cfg(test)]
        let mut live_count = 0u32;
        #[cfg(test)]
        for node in &nodes {
            if node.enabled {
                enabled_count = enabled_count.saturating_add(1);
            }
        }

        let mut needed = FrameResource::FINAL_OUTPUT;
        if input.run_selection_reduce {
            needed = needed.union(FrameResource::SELECTION_READBACK);
        }

        // Backward reachability pass in reverse topological order.
        for node in nodes.iter().rev() {
            if !node.enabled {
                continue;
            }
            if node.writes.intersects(needed) {
                if let Some(slot) = live.get_mut(node.id as usize) {
                    *slot = true;
                }
                #[cfg(test)]
                {
                    live_count = live_count.saturating_add(1);
                }
                needed = needed.union(node.reads);
            }
        }

        let is_live = |id: PassId| live.get(id as usize).copied().unwrap_or(false);

        let depth_needed_after_phase2 = is_live(PassId::PostOpaqueHooks)
            || is_live(PassId::SceneTransparent)
            || is_live(PassId::PostSceneResolvedHooks);
        let keep_depth_after_primary = is_live(PassId::HiZBuild) || depth_needed_after_phase2;
        let keep_depth_after_phase2 = depth_needed_after_phase2;

        let msaa_needed_after_phase2 = is_live(PassId::PostOpaqueHooks)
            || is_live(PassId::SceneTransparent)
            || is_live(PassId::SceneResolve)
            || (input.continuous_second_pass && !is_live(PassId::PostSceneResolvedHooks));
        let keep_msaa_after_primary = is_live(PassId::ScenePhase2) || msaa_needed_after_phase2;
        let keep_msaa_after_phase2 = msaa_needed_after_phase2;

        Self {
            live,
            #[cfg(test)]
            enabled_count,
            #[cfg(test)]
            live_count,
            keep_depth_after_primary,
            keep_depth_after_phase2,
            keep_msaa_after_primary,
            keep_msaa_after_phase2,
            separate_transparent_pass,
        }
    }

    #[inline]
    pub(crate) fn is_live(&self, id: PassId) -> bool {
        self.live.get(id as usize).copied().unwrap_or(false)
    }

    #[inline]
    pub(crate) const fn keep_depth_after_primary(&self) -> bool {
        self.keep_depth_after_primary
    }

    #[inline]
    pub(crate) const fn keep_depth_after_phase2(&self) -> bool {
        self.keep_depth_after_phase2
    }

    #[inline]
    pub(crate) const fn keep_msaa_after_primary(&self) -> bool {
        self.keep_msaa_after_primary
    }

    #[inline]
    pub(crate) const fn keep_msaa_after_phase2(&self) -> bool {
        self.keep_msaa_after_phase2
    }

    #[inline]
    pub(crate) const fn separate_transparent_pass(&self) -> bool {
        self.separate_transparent_pass
    }

    #[cfg(test)]
    pub(crate) const fn pruned_count(&self) -> u32 {
        self.enabled_count.saturating_sub(self.live_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_single_pass_discards_depth_and_prunes_optional_stages() {
        let plan = PassGraph::compile(FramePlanInput {
            rebuild_scene: true,
            msaa_samples: 4,
            ..FramePlanInput::default()
        });
        assert!(plan.is_live(PassId::ScenePhase1));
        assert!(!plan.is_live(PassId::PreSceneHooks));
        assert!(!plan.is_live(PassId::HiZBuild));
        assert!(!plan.is_live(PassId::PostOpaqueHooks));
        assert!(!plan.is_live(PassId::PostSceneResolvedHooks));
        assert!(!plan.is_live(PassId::DynamicComposite));
        assert!(!plan.is_live(PassId::SelectionReduce));
        assert!(!plan.keep_depth_after_primary());
        assert!(!plan.keep_msaa_after_primary());
        assert_eq!(plan.pruned_count(), 0);
    }

    #[test]
    fn post_opaque_and_post_resolved_hooks_preserve_depth_and_split_transparent() {
        let plan = PassGraph::compile(FramePlanInput {
            rebuild_scene: true,
            has_stencil: true,
            has_post_opaque_hook: true,
            has_transparent_layers: true,
            has_post_scene_resolved_hook: true,
            msaa_samples: 4,
            ..FramePlanInput::default()
        });
        assert!(plan.is_live(PassId::ScenePhase1));
        assert!(plan.is_live(PassId::PostOpaqueHooks));
        assert!(plan.is_live(PassId::SceneTransparent));
        assert!(!plan.is_live(PassId::SceneResolve));
        assert!(plan.is_live(PassId::PostSceneResolvedHooks));
        assert!(plan.keep_depth_after_primary());
        assert!(plan.keep_msaa_after_primary());
        assert!(plan.separate_transparent_pass());
    }

    #[test]
    fn post_opaque_without_transparent_schedules_explicit_msaa_resolve() {
        let plan = PassGraph::compile(FramePlanInput {
            rebuild_scene: true,
            has_post_opaque_hook: true,
            has_transparent_layers: false,
            msaa_samples: 4,
            ..FramePlanInput::default()
        });
        assert!(plan.is_live(PassId::PostOpaqueHooks));
        assert!(!plan.is_live(PassId::SceneTransparent));
        assert!(plan.is_live(PassId::SceneResolve));
        assert!(plan.keep_msaa_after_primary());
    }

    #[test]
    fn cached_on_demand_overlay_skips_scene_and_prunes_outline_when_unused() {
        let plan = PassGraph::compile(FramePlanInput {
            rebuild_scene: false,
            need_dynamic: true,
            has_outline: true,
            msaa_samples: 4,
            ..FramePlanInput::default()
        });
        assert!(!plan.is_live(PassId::ScenePhase1));
        assert!(plan.is_live(PassId::OutlineMask));
        assert!(plan.is_live(PassId::OutlineJfa));
        assert!(plan.is_live(PassId::DynamicComposite));
    }

    #[test]
    fn two_phase_gpu_cull_keeps_depth_for_hiz_and_discards_after_phase2() {
        let plan = PassGraph::compile(FramePlanInput {
            rebuild_scene: true,
            use_gpu_cull: true,
            msaa_samples: 4,
            ..FramePlanInput::default()
        });
        assert!(plan.is_live(PassId::GpuCullPhase1));
        assert!(plan.is_live(PassId::ScenePhase1));
        assert!(plan.is_live(PassId::HiZBuild));
        assert!(plan.is_live(PassId::GpuCullPhase2));
        assert!(plan.is_live(PassId::ScenePhase2));
        assert!(plan.keep_depth_after_primary());
        assert!(!plan.keep_depth_after_phase2());
        assert!(plan.keep_msaa_after_primary());
        assert!(!plan.keep_msaa_after_phase2());
    }
}
