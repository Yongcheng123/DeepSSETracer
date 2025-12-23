# vim: set expandtab shiftwidth=4 softtabstop=4:

from chimerax.core.toolshed import BundleAPI


class _DLAPI(BundleAPI):
    """ChimeraX bundle API for DeepSSETracer."""

    api_version = 1

    @staticmethod
    def start_tool(session, bi, ti):
        """
        Initialize and return tool instance.
        
        Args:
            session: ChimeraX session instance
            bi: BundleInfo instance
            ti: ToolInfo instance
        """
        if ti.name == "DeepSSETracer":
            from . import tool
            return tool.DLStruct(session, ti.name)
        raise ValueError(f"Unknown tool: {ti.name}")

    @staticmethod
    def get_class(class_name):
        """Return tool class by name."""
        if class_name == "DeepSSETracer":
            from . import tool
            return tool.DLStruct
        raise ValueError(f"Unknown class: {class_name}")


bundle_api = _DLAPI()
