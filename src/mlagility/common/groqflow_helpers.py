"""
Helper functions for interfacing with GroqFlow
"""

import groqflow.common.build as build
from groqflow.justgroqit.stage import GroqitStage


class SuccessStage(GroqitStage):
    """
    Stage that sets state.build_status = build.Status.SUCCESSFUL_BUILD,
    indicating to groqit() that the build can be used for benchmarking
    CPUs and GPUs.
    """

    def __init__(self):
        super().__init__(
            unique_name="set_success",
            monitor_message="Finishing up",
        )

    def fire(self, state: build.State):
        state.build_status = build.Status.SUCCESSFUL_BUILD

        return state
