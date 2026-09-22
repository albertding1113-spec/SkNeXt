from brainglobe_atlasapi import BrainGlobeAtlas
import tifffile


if __name__ =="__main__":
    # Download/open Allen CCFv3 at 10 µm
    atlas = BrainGlobeAtlas("allen_mouse_10um")

    # Bregma -1.19 mm
    bregma_ap = -1.19  # mm

    # Chon/SHARP-Track convention:
    # CCF AP = -Bregma_AP * 1000 + 5400
    ccf_ap_um = -bregma_ap * 1000 + 5400

    resolution = 10  # µm
    ap_index = round(ccf_ap_um / resolution)

    print("CCF AP:", ccf_ap_um, "µm")
    print("AP slice index:", ap_index)

    # Extract coronal slice
    template_slice = atlas.template[ap_index, :, :]
    annotation_slice = atlas.annotation[ap_index, :, :]

    # Save as TIFF
    tifffile.imwrite(
        r"E:\Albert_BigFile\Data\260618_SkNeXt_dataset\CCFv3_bregma_-1.19_template.tif",
        template_slice
    )

    tifffile.imwrite(
        r"E:\Albert_BigFile\Data\260618_SkNeXt_dataset\CCFv3_bregma_-1.19_annotation.tif",
        annotation_slice
    )