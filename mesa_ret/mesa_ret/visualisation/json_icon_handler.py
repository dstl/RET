"""JSON Writer icon handling."""
import os
import pathlib
import shutil
import warnings


class IconCopier:
    """A class to process icon copying."""

    icon_list: list[str] = []

    output_folder_name: str = "output"

    def copy_icon(self, icon_path: str) -> str:
        """Copy a given icon and return the name.

        If an icon has not yet been copied into the assets folder then it will first be copied.
        Then the name will be returned. One limitation of this method is that icons with the
        same filename but originally in different folders will be ignored.

        Args:
            icon_path (str): the absolute path of the icon before copying.

        Returns:
            str: name of the copied icon.
        """
        path = pathlib.Path(icon_path)
        icon_name = path.name
        if path.exists() and path.is_file():
            if icon_name not in self.icon_list:

                relative_path = f"{self.output_folder_name}/assets/icons/{path.name}"
                copy_path = os.path.abspath(relative_path)
                if not pathlib.Path(copy_path).is_file():
                    os.makedirs(os.path.dirname(copy_path), exist_ok=True)
                    shutil.copy(path, copy_path)

                self.icon_list.append(icon_name)
            return icon_name
        warnings.warn("icon path either doesn't exist or is not a file.")
        return icon_path
