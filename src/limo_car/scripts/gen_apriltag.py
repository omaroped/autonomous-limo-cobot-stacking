#!/usr/bin/python3
"""
gen_apriltag.py — Generate AprilTag 36h11 marker PNGs for the place table.

Run once (no ROS needed):
    python3 gen_apriltag.py

Writes:
    ../media/materials/textures/apriltag_<id>.png  — white-border marker (512×512 px)
    ../media/materials/scripts/apriltag.material    — OGRE material per tag ID

Four tags, one per table face (distinct IDs so the estimator can identify which
face it sees and compute the correct table centre direction):
    ID 0 → North face  (tag facing +Y)
    ID 1 → South face  (tag facing -Y)
    ID 2 → East face   (tag facing +X)
    ID 3 → West face   (tag facing -X)
"""

import cv2
import cv2.aruco as aruco
import numpy as np
import os

TAG_IDS   = [0, 1, 2, 3]
TAG_NAMES = ['north', 'south', 'east', 'west']
IMG_SIZE  = 512      # pixels
BORDER_PX = 48      # white border around the tag (required by ArUco detector)
DICT_ID   = aruco.DICT_APRILTAG_36H11   # cv2 4.5.x uses capital H variant

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TEXTURE_DIR  = os.path.join(SCRIPT_DIR, '..', 'media', 'materials', 'textures')
MATERIAL_DIR = os.path.join(SCRIPT_DIR, '..', 'media', 'materials', 'scripts')


def generate_pngs():
    dictionary = aruco.getPredefinedDictionary(DICT_ID)
    os.makedirs(TEXTURE_DIR, exist_ok=True)

    for tag_id, name in zip(TAG_IDS, TAG_NAMES):
        # cv2 4.5.x API: drawMarker(dict, id, sidePixels) → greyscale img
        marker = aruco.drawMarker(dictionary, tag_id, IMG_SIZE - 2 * BORDER_PX)
        canvas = np.ones((IMG_SIZE, IMG_SIZE), dtype=np.uint8) * 255
        canvas[BORDER_PX:BORDER_PX + marker.shape[0],
               BORDER_PX:BORDER_PX + marker.shape[1]] = marker
        # Convert to RGBA (Gazebo OGRE needs PNG with alpha for texture mapping)
        rgba = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGRA)
        out = os.path.join(TEXTURE_DIR, f'apriltag_{tag_id}.png')
        cv2.imwrite(out, rgba)
        print(f'  wrote {out}  (ID={tag_id} face={name})')


def generate_material():
    os.makedirs(MATERIAL_DIR, exist_ok=True)
    lines = []
    for tag_id in TAG_IDS:
        tex_name = f'apriltag_{tag_id}.png'
        mat_name = f'AprilTag_{tag_id}'
        lines.append(f'material {mat_name}')
        lines.append('{')
        lines.append('    technique')
        lines.append('    {')
        lines.append('        pass')
        lines.append('        {')
        lines.append('            lighting off')
        lines.append('            texture_unit')
        lines.append('            {')
        lines.append(f'                texture {tex_name}')
        lines.append('                filtering none')
        lines.append('            }')
        lines.append('        }')
        lines.append('    }')
        lines.append('}')
        lines.append('')
    out = os.path.join(MATERIAL_DIR, 'apriltag.material')
    with open(out, 'w') as f:
        f.write('\n'.join(lines))
    print(f'  wrote {out}')


if __name__ == '__main__':
    print('Generating AprilTag 36h11 markers…')
    generate_pngs()
    generate_material()
    print('Done.')
