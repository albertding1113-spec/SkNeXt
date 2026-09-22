from pathlib import Path
from PIL import Image


def black_to_transparent(
    input_folder,
    output_folder=None,
    threshold=10,
    overwrite=False,
):
    """
    Convert black / near-black pixels in all PNG images to transparent.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing PNG images.

    output_folder : str or Path, optional
        Folder for processed images.
        If None, creates "<input_folder>_transparent".

    threshold : int
        RGB values <= threshold are considered black.
        Example:
            threshold=0   -> only pure black [0, 0, 0]
            threshold=10  -> very dark pixels also become transparent
            threshold=30  -> more aggressive background removal

    overwrite : bool
        If True, overwrite the original PNG files.
    """

    input_folder = Path(input_folder)

    if overwrite:
        output_folder = input_folder
    else:
        if output_folder is None:
            output_folder = input_folder.parent / f"{input_folder.name}_transparent"
        else:
            output_folder = Path(output_folder)

        output_folder.mkdir(parents=True, exist_ok=True)

    png_files = list(input_folder.glob("*.png"))

    print(f"Found {len(png_files)} PNG files.")

    for i, file_path in enumerate(png_files, start=1):

        # Convert to RGBA so the image has an alpha channel
        img = Image.open(file_path).convert("RGBA")

        pixels = img.load()
        width, height = img.size

        for y in range(height):
            for x in range(width):
                r, g, b, a = pixels[x, y]

                # Black / near-black -> fully transparent
                if r <= threshold and g <= threshold and b <= threshold:
                    pixels[x, y] = (r, g, b, 0)

        output_path = output_folder / file_path.name
        img.save(output_path)

        print(f"[{i}/{len(png_files)}] Saved: {output_path}")

    print("Finished.")


if __name__ == "__main__":

    input_folder = r"D:\Albert\Figure\ZhaoLab\SkNeXt\260907_snapshot_for_SkNeXt"

    black_to_transparent(
        input_folder=input_folder,
        threshold=5,
        overwrite=False,
    )