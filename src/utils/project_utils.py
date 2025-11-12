from pathlib import Path

class ProjectUtils:
    """
    Utility class for project-related operations.
    """
    
    @staticmethod
    def get_project_root() -> Path:
        """
        Returns the project root directory.
        :return: Path object representing the project root directory.
        """
        return Path(__file__).resolve().parent.parent.parent
