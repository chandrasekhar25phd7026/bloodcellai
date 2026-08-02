"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    base_transform.py

Version:
    1.0.0

Description
-----------
Abstract base class for all preprocessing transforms.

All preprocessing transforms must inherit from BaseTransform.

Responsibilities
----------------
✓ Common interface
✓ Configuration management
✓ Timing
✓ Error handling
✓ Transform history
✓ Logging support
✓ Extensible architecture

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter
from typing import Any

from preprocessing.preprocessing_models import (
    TransformRecord,
    TransformStatus,
)

from preprocessing.preprocessing_config import (
    PreprocessingConfig,
)


# =============================================================================
# Base Transform
# =============================================================================

class BaseTransform(ABC):
    """
    Abstract base class for all preprocessing transforms.
    """

    def __init__(
        self,
        config: PreprocessingConfig,
    ) -> None:

        self._config = config

        self._enabled = True

        self._name = self.__class__.__name__

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def name(self) -> str:
        """
        Name of the preprocessing transform.
        """

        return self._name

    @property
    def config(self) -> PreprocessingConfig:
        """
        Return preprocessing configuration.
        """

        return self._config

    @property
    def enabled(self) -> bool:
        """
        Returns True if transform is enabled.
        """

        return self._enabled

    @enabled.setter
    def enabled(
        self,
        value: bool,
    ) -> None:

        self._enabled = value

    # -------------------------------------------------------------------------
    # Abstract API
    # -------------------------------------------------------------------------

    @abstractmethod
    def apply(
        self,
        image: Any,
    ) -> Any:
        """
        Apply preprocessing transform.

        Must be implemented by subclasses.
        """
        raise NotImplementedError
    # -------------------------------------------------------------------------
    # Execution
    # -------------------------------------------------------------------------

    def __call__(
        self,
        image: Any,
    ) -> tuple[Any, TransformRecord]:
        """
        Execute the transform.

        Returns
        -------
        tuple
            (processed_image, TransformRecord)
        """

        if not self.enabled:

            record = TransformRecord(
                name=self.name,
                status=TransformStatus.SKIPPED,
                message="Transform disabled."
            )

            return image, record

        start_time = perf_counter()

        try:

            self.validate_input(image)

            self.before_apply(image)

            processed_image = self.apply(image)

            self.validate_output(processed_image)

            self.after_apply(processed_image)

            execution_time = (
                perf_counter() - start_time
            )

            record = TransformRecord(

                name=self.name,

                status=TransformStatus.SUCCESS,

                execution_time=execution_time,

                parameters=self.parameters(),

                message="Success"

            )

            return processed_image, record

        except Exception as ex:

            execution_time = (
                perf_counter() - start_time
            )

            record = TransformRecord(

                name=self.name,

                status=TransformStatus.FAILED,

                execution_time=execution_time,

                parameters=self.parameters(),

                message=str(ex)

            )

            if self.config.pipeline.raise_exceptions:
                raise

            return image, record

    # -------------------------------------------------------------------------
    # Configuration
    # -------------------------------------------------------------------------

    def parameters(self) -> dict[str, Any]:
        """
        Return transform parameters.

        Subclasses should override this method if they
        want their parameters recorded.
        """

        return {}

    # -------------------------------------------------------------------------
    # Hooks
    # -------------------------------------------------------------------------

    def before_apply(
        self,
        image: Any,
    ) -> None:
        """
        Hook executed before apply().
        """

        return None

    def after_apply(
        self,
        image: Any,
    ) -> None:
        """
        Hook executed after apply().
        """

        return None
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------

    def validate_input(
        self,
        image: Any,
    ) -> None:
        """
        Validate the input image before processing.

        Subclasses may override this method to perform
        additional validation.
        """

        if image is None:
            raise ValueError(
                "Input image cannot be None."
            )

    def validate_output(
        self,
        image: Any,
    ) -> None:
        """
        Validate the processed image.

        Subclasses may override this method.
        """

        if image is None:
            raise ValueError(
                "Processed image cannot be None."
            )

    # -------------------------------------------------------------------------
    # State Management
    # -------------------------------------------------------------------------

    def reset(self) -> None:
        """
        Reset transform state.

        Override in subclasses if internal state exists.
        """

        return None

    # -------------------------------------------------------------------------
    # Logging Hooks
    # -------------------------------------------------------------------------

    def log(
        self,
        message: str,
    ) -> None:
        """
        Logging hook.

        Override this method to integrate with the
        framework logger.
        """

        if self.config.pipeline.log_each_transform:
            print(f"[{self.name}] {message}")

    # -------------------------------------------------------------------------
    # Information
    # -------------------------------------------------------------------------

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}("
            f"enabled={self.enabled})"
        )

    def __str__(self) -> str:

        return self.__class__.__name__