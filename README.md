# AIRL_Coding_Assignment

Text-Driven Image & Video Segmentation with SAM 2

This project demonstrates a powerful pipeline for performing segmentation on both static images and video clips using natural language text prompts. The system leverages the zero-shot object detection capabilities of GroundingDINO to interpret text and locate objects, and the high-quality segmentation power of the Segment Anything Model 2 (SAM 2) to generate precise masks.
🌟 Core Technologies

    GroundingDINO: A state-of-the-art, open-set object detector that can locate arbitrary objects in an image based on a free-text query. It acts as the "eyes" of our pipeline, identifying where the object of interest is.

    Segment Anything Model 2 (SAM 2): The successor to the original SAM, SAM 2 is a foundation model for image segmentation. It can generate high-quality masks for objects given various prompts, including points, boxes, or even other masks. In this pipeline, it takes the bounding boxes from GroundingDINO to determine the exact pixel-level boundaries of the object.

⚙️ The Pipeline: From Text to Mask

The core of this project is a two-stage process that seamlessly translates a user's text prompt into a high-fidelity segmentation mask.
1. Image Segmentation

For a single image, the workflow is as follows:

    Input: The process starts with two inputs: an image and a text prompt (e.g., "the black cat sitting on the couch").

    Region Seeding (GroundingDINO):

        The input image and the text prompt are fed into the GroundingDINO model.

        GroundingDINO processes this information and outputs a set of bounding box coordinates that correspond to the object(s) described in the text. This step essentially converts the abstract text query into concrete spatial locations (region seeds) within the image.

    Preprocessing for SAM 2:

        Before being used by SAM 2, the image and the bounding box prompts from GroundingDINO undergo necessary preprocessing.

        Image Preparation: The input image is loaded and converted from the BGR color space (used by OpenCV) to the RGB space, which is the standard format expected by SAM 2's internal encoder. The SAM2ImagePredictor then handles the final resizing and normalization.

        Bounding Box Conversion: The bounding boxes from GroundingDINO are provided in a normalized [center_x, center_y, width, height] format. They are converted to the absolute [x1, y1, x2, y2] pixel coordinate format that SAM 2 requires for its prompts.

    Mask Generation (SAM 2):

        The preprocessed image is set in the SAM 2 predictor, which calculates its image embedding.

        The properly formatted bounding boxes are then passed as prompts to the predictor.

        SAM 2 uses these boxes to "focus" on the specified regions and generates precise, pixel-perfect segmentation masks for the objects contained within them.

    Final Output:

        The resulting mask is overlaid onto the original image, visually highlighting the segmented object.

This entire process is end-to-end, requiring no manual annotation or model training for new objects.
2. Video Segmentation (Bonus Extension)

The pipeline is extended to handle video clips by propagating masks across frames, creating a simple yet effective object tracking and segmentation system.

    Initialization (First Frame):

        The first frame of the video is processed exactly like a single image. GroundingDINO uses the text prompt to generate an initial bounding box for the object of interest, and SAM 2 creates the first mask.

    Propagation (Subsequent Frames):

        For every subsequent frame in the video, we no longer need GroundingDINO. Instead, we use the output from the previous frame to guide the segmentation of the current frame.

        The bounding box of the mask generated for frame N-1 is calculated.

        This bounding box is then used as the prompt for SAM 2 to segment the object in frame N.

    Output Video:

        This process repeats for the entire video. Each frame is annotated with the generated mask, and the frames are stitched back together to create a final output video where the target object is continuously segmented.

⚠️ Limitations and Considerations

While effective, this approach has several inherent limitations:

    Dependency on Initial Detection: The final segmentation quality is heavily reliant on the performance of GroundingDINO. If the initial text-to-box detection is inaccurate, noisy, or fails entirely, SAM 2 will produce a poor or incorrect mask.

    Simple Tracking Logic: The video propagation method is naive. It can easily fail if:

        The object moves too quickly between frames.

        The object is temporarily occluded by another object.

        There are multiple similar-looking objects in the scene, which could cause the tracking to "jump" to the wrong one.

    No Object Re-identification: If the tracker loses the object, it has no mechanism to re-identify it later in the video. The text prompt is only used once on the initial frame.

    Computational Resources: Both GroundingDINO and SAM 2 are large, computationally intensive models. Processing high-resolution images, and especially videos, requires a capable GPU for reasonable performance. The entire pipeline is designed to be runnable in a Google Colab environment with a GPU runtime.
