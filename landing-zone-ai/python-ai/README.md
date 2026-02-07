# Python AI Service

## Dataset Setup (WildUAV)

1. **Download**:
   - Go to the [WildUAV GitHub Repository](https://github.com/ewrfCASGX/WildUAV) (or search for it) and download the **Mapping Set**.
   - You need the RGB images (`seqXX_img+metadata.7z`).
   - (Optional) Download the Depth maps (`seqXX_depth.7z`) if you plan to use depth data.

2. **Extract**:
   - Create a directory `python-ai/data/WildUAV`.
   - Extract the `Mapping` folder into it.
   - Final structure should look like:
     ```
     python-ai/
       data/
         WildUAV/
           Mapping/
             seq00/
               img/
               depth/
             seq01/
             ...
     ```

3. **Preprocess (Resize)**:
   - The original images are very large (5280x3956). To resize them for training/inference (e.g., to 512x512):
   ```bash
   # From python-ai directory
   python -m scripts.prepare_dataset --root data/WildUAV --output data/WildUAV_Processed --width 512 --height 512
   ```
