"""Read video files into the frame arrays Model.run expects."""
import numpy as np


def read_video(path, max_frames=None, size=None):
    """-> (frames uint8 [F, H, W, 3], fps). size=(width, height) downsamples on read, which saves memory for long videos."""
    import imageio.v2 as imageio
    kw = dict(size=size) if size else {}
    rd = imageio.get_reader(path, 'ffmpeg', **kw); fps = rd.get_meta_data().get('fps', 30)
    out = []
    for i, f in enumerate(rd):
        if max_frames is not None and i >= max_frames: break
        out.append(np.asarray(f)[..., :3])
    rd.close()
    return np.stack(out), float(fps)
