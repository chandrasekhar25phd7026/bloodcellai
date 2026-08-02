"""
===============================================================================
BloodCellAI Preprocessing Framework
===============================================================================

File:
    registry.py

Version:
    1.0.0

Description
-----------
Registry for preprocessing transforms.

Responsibilities
----------------
✓ Register transforms
✓ Retrieve transforms
✓ Create transform instances
✓ List available transforms
✓ Extensible plugin architecture

Author:
    Sekhar Muthangi
===============================================================================
"""

from __future__ import annotations

from typing import Type

from .base_transform import BaseTransform
from preprocessing.preprocessing_config import PreprocessingConfig


# =============================================================================
# Transform Registry
# =============================================================================

class TransformRegistry:
    """
    Registry for all preprocessing transforms.

    This class maintains a mapping between transform names
    and their corresponding classes.
    """

    _registry: dict[str, Type[BaseTransform]] = {}

    # -------------------------------------------------------------------------
    # Registration
    # -------------------------------------------------------------------------

    @classmethod
    def register(
        cls,
        name: str,
        transform_class: Type[BaseTransform],
    ) -> None:
        """
        Register a preprocessing transform.

        Parameters
        ----------
        name
            Unique transform name.

        transform_class
            Class derived from BaseTransform.
        """

        if not issubclass(transform_class, BaseTransform):
            raise TypeError(
                f"{transform_class.__name__} "
                "must inherit from BaseTransform."
            )

        key = name.lower().strip()

        if key in cls._registry:
            raise ValueError(
                f"Transform '{key}' is already registered."
            )

        cls._registry[key] = transform_class

    # -------------------------------------------------------------------------
    # Removal
    # -------------------------------------------------------------------------

    @classmethod
    def unregister(
        cls,
        name: str,
    ) -> None:
        """
        Remove a registered transform.
        """

        cls._registry.pop(
            name.lower().strip(),
            None,
        )
    # -------------------------------------------------------------------------
    # Lookup
    # -------------------------------------------------------------------------

    @classmethod
    def get(
        cls,
        name: str,
    ) -> type[BaseTransform]:
        """
        Retrieve a registered transform class.

        Parameters
        ----------
        name
            Transform name.

        Returns
        -------
        type[BaseTransform]
            Registered transform class.

        Raises
        ------
        KeyError
            If the transform is not registered.
        """

        key = name.lower().strip()

        if key not in cls._registry:
            raise KeyError(
                f"Transform '{key}' is not registered."
            )

        return cls._registry[key]

    @classmethod
    def exists(
        cls,
        name: str,
    ) -> bool:
        """
        Check whether a transform is registered.
        """

        return (
            name.lower().strip()
            in cls._registry
        )

    # -------------------------------------------------------------------------
    # Information
    # -------------------------------------------------------------------------

    @classmethod
    def available_transforms(
        cls,
    ) -> list[str]:
        """
        Return a sorted list of registered transform names.
        """

        return sorted(
            cls._registry.keys()
        )

    @classmethod
    def registry(
        cls,
    ) -> dict[str, type[BaseTransform]]:
        """
        Return a copy of the transform registry.
        """

        return cls._registry.copy()

    @classmethod
    def count(
        cls,
    ) -> int:
        """
        Return the number of registered transforms.
        """

        return len(cls._registry)

    # -------------------------------------------------------------------------
    # Maintenance
    # -------------------------------------------------------------------------

    @classmethod
    def clear(
        cls,
    ) -> None:
        """
        Remove all registered transforms.
        """

        cls._registry.clear()
    # -------------------------------------------------------------------------
    # Factory
    # -------------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        config: PreprocessingConfig,
    ) -> BaseTransform:
        """
        Create a transform instance.

        Parameters
        ----------
        name
            Registered transform name.

        config
            Preprocessing configuration.

        Returns
        -------
        BaseTransform
            Instantiated transform.
        """

        transform_class = cls.get(name)

        return transform_class(config)

    # -------------------------------------------------------------------------
    # Special Methods
    # -------------------------------------------------------------------------

    def __contains__(
        self,
        name: str,
    ) -> bool:
        """
        Support:

            "resize" in registry
        """

        return self.exists(name)

    def __len__(self) -> int:
        """
        Return number of registered transforms.
        """

        return self.count()

    def __repr__(self) -> str:

        return (
            f"{self.__class__.__name__}"
            f"(registered={self.count()})"
        )

    def __str__(self) -> str:

        return (
            f"{self.count()} registered transforms"
        )