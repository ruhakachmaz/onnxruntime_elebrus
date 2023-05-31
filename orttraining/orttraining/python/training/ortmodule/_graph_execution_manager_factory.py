# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from typing import Union

from ._fallback import _FallbackManager
from ._inference_manager import InferenceManager
from ._io import _FlattenedModule
from ._training_manager import TrainingManager
from .debug_options import DebugOptions


class GraphExecutionManagerFactory:
    def __init__(self, module: _FlattenedModule, debug_options: DebugOptions, fallback_manager: _FallbackManager):
        self._training_manager = TrainingManager(module, debug_options, fallback_manager)
        self._inference_manager = InferenceManager(module, debug_options, fallback_manager)

    def __call__(self, is_training) -> Union[InferenceManager, TrainingManager]:
        if is_training:
            return self._training_manager
        else:
            return self._inference_manager
