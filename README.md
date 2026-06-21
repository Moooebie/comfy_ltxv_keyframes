# comfy_ltxv_keyframes
Supports specifying the last frame in addition to the first for video generation with LTXV in Comfy.


This is a work based on the `comfy/comfy_lt.py` file in the original ComfyUI codebase.

## Usage
1. Place the file under the `custom_nodes` directory inside your ComfyUI instance.
2. (Re)start ComfyUI, find the `LTXVImgToVideoStartEnd` node, and add it.
3. Load an image and connect it to the optional `imasge_last` input, so it will be used to specify the ending frame of the video.

If no last frame image is provided, the node behaves exactly the same as the `LTXVImgToVideoInplace` node.

<img width="799" height="591" alt="image" src="https://github.com/user-attachments/assets/7011996e-e73e-4d49-9fd2-513c0c3cbbec" />
