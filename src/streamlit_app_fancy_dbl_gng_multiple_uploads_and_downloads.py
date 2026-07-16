import streamlit as st
import numpy as np
import constants
import dbl_gng
import clustering
import color_pca
from tqdm import trange
from PIL import Image, ImageDraw
from torchvision import transforms
import matplotlib.pyplot as plt
import io, zipfile
import datetime
import fancy_pca as FP
import time



#----------------------------UI Constants-------------------------------------------------------
#Name constant for fancy gng
FANCYGNG_STR = "Fancy GNG"
#Name constant for fancy pca
FANCYPCA_STR = "Fancy PCA"
#Name constant for color jitter
COLORJITTER_STR = "Color Jitter"
#Max number of generated preview images in the ui 
MAX_UI_AUG_COUNT = 10
MAX_UI_AUG_COUNT += 1
#Basic point cloud size 
CLOUD_SIZE = 1000
#Basic PC1/PC2 size
PC12_SIZE = 1000
#Basic point cluster cloud size 
CLUSTER_CLOUD_SIZE = 1000
#Basic number of datapoints for reduced fancy gng training
REDUCED_TRAINING = 1000
#Margin in the subplot between the rows and column
SUB_PLOT_MARGIN = 7

# Plot font / layout
TITLE_FONT_SIZE = 30
TICK_LABEL_SIZE = 16
TITLE_PAD = 16

# Separate title spacing for image-like plots
IMAGE_TITLE_PAD = 10
IMAGE_TITLE_Y = 1.02

# Separate title spacing for point/scatter plots
PLOT_TITLE_PAD = TITLE_PAD
PLOT_TITLE_Y = 1.02

#-----------------------------Session------------------------------------------------------------

#Init new session variables
def init_session():
    # session states 
    #save uploaded files in session
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = None
    #save generated images in session
    if "image_results" not in st.session_state:
        st.session_state.image_results = {}  # {filename: {"original": Image, "aug_images": [...], "cluster_count": int, "data_shape": tuple}}
    #save ui figure in session
    if "fig_png" not in st.session_state:
        st.session_state.fig_png = {}
    #save state of augmentation in session
    if "done" not in st.session_state:
        st.session_state.done = False
    #save camera picture in session
    if "last_picture" not in st.session_state:
        st.session_state.last_picture = None
    #save last augmentation technique in session
    if "last_aug" not in st.session_state:
        st.session_state.last_aug = None
    #save ui augmentation information in session
    if "last_aug_info" not in st.session_state:
        st.session_state.last_aug_info = None
    #save gray scale images in session
    if "gray_images" not in st.session_state:
        st.session_state.gray_images = {}

init_session()

#reset complete session variables 
def reset_session():
    st.session_state.clear()
    init_session()

#reset session variables for a new augmentation
def reset_for_new_run():
    st.session_state.image_results = {}
    st.session_state.fig_png = {}
    st.session_state.gray_images = {}



#--------------------------Streamlit UI-------------------------------
st.title("🧠 Fancy GNG image augmentation")
st.write("Upload one or more images or take one with your camera.")


# Choose augmentation technique
aug_option = st.selectbox(
    "Select the augmentation method:",
    [FANCYGNG_STR, FANCYPCA_STR, COLORJITTER_STR],
    index=0,
    help="Select the augmentation method for generating the images."
)
st.write(f"Method chosen: {aug_option}")


# Choose image source
input_option = st.radio("Select image source:", ["File upload", "Camera"],
                help="Select the source of the images to be used for augmentation. " \
                "When selecting the camera, only one image can be captured. When selecting file upload, multiple images can be selected.")

# File upload
if input_option == "File upload":
    uploaded_files = st.file_uploader(
        "Select images", 
        type=["jpg", "jpeg", "png"], 
        accept_multiple_files=True,
        on_change= reset_session,
        help="Multiple images are accepted. The supported types are: jpg, jpeg and png"
    )

    if uploaded_files:
        #save images in session
        st.session_state.uploaded_files = uploaded_files

# Camera input
elif input_option == "Camera":
    camera_image = st.camera_input("Take a picture",
    help="A webcam is required to take a picture.")

    if camera_image is not None:

        #If the previous augmentation was based on uploaded files
        if st.session_state.last_picture is None:
            reset_session()

        #If a new picture was taken
        elif st.session_state.last_picture is not None and camera_image.getvalue() != st.session_state.last_picture:
            reset_session()
        
        #timestamp for image name
        timestamp = datetime.datetime.now().strftime("%Y%m%d")
        camera_image.name = f"camera_{timestamp}.jpg"        
        st.session_state.uploaded_files = [camera_image]
        st.session_state.last_picture = camera_image.getvalue()      
        
#Save activated buttons which have an impact on the amount of generated rows in the preview figure    
option_buttons_ui = []  

#Option: Generate preview figures
figures = st.checkbox("Generate visualization",
                value = True,
                help="Generate a quick view of the result. If false, the images are still available for download. No visualization is useful when larger data sets are to be augmented")



#Option: Generate gray scale
show_gray_scale = st.checkbox("Additionally generate a grayscale version",
                    help="Additionally a grayscale version of the augmented image is generated. These images can also be downloaded at the end.")


#Basic init for additional visualization
show_cluster = False
reduced_fancy_gng = False
show_cluster_cloud = False
use_original_size_pc12 = False
show_cluster_pc12_ev = False

#Augmentation technique: Fancy gng
if aug_option == FANCYGNG_STR:
   
    #Reduce fancy gng training on fewer datapoints
    reduced_fancy_gng = st.checkbox("Train Fancy GNG on fewer data points",
                    help="At random (without repetition), select n pixels from the image to be used to train the GNG. " \
                    "This can be used to accelerate Fancy GNG.")
    
    #Additional fancy gng visualization 
    if figures:
        #Generate pixel cluster mapping visualization
        show_cluster = st.checkbox("Generate a pixel cluster map",
                        help="A unique color is selected for each color cluster (connected codebook vectors) found by GNG. " \
                        "All pixels belonging to this cluster are colored in this color. This visualizes the color clusters found in the image.")
        
        #Generate the cluster point cloud visualization
        show_cluster_cloud = st.checkbox("Show the cluster cloud",
                        help="This option generates a point cloud representation of each image's pixels in the G-B color space. " \
                        "Each point represents one pixel of the image. The position of the points is based on the green and blue values (x=G, y=B). " \
                        "Every point is colored in their corresponding cluster group color. " \
                        "The result is a 2D visual representation of the color distribution of an image.")   
        option_buttons_ui.append(show_cluster_cloud)
 

show_point_cloud = False
#Option: Generate image point cloud
if figures:
    show_point_cloud = st.checkbox("Show the point cloud",
                        help="This option generates a point cloud representation of each image's pixels in the G-B color space. " \
                        "Each point represents one pixel of the image.The position of the points is based on the green and blue values (x=G, y=B). " \
                        "The color of the points corresponds to the original color of the pixel (R,G,B). " \
                        "The result is a 2D visual representation of the color distribution of an image.")   
    option_buttons_ui.append(show_point_cloud)

#Option: Generate image PC1/PC2
show_pc12 = False
if figures:
    show_pc12 = st.checkbox("Show the PC1/PC2 plot",
                        help="This option generates a 2D PCA projection of each image's pixels in RGB color space using the principal components of the original image. " \
                        "Each point represents one pixel and is colored with its original RGB value. " \
                        "This allows direct comparison of the color distribution of the original and augmented images in the same PCA basis.")
    option_buttons_ui.append(show_pc12)

option_buttons_ui.append(show_cluster_cloud and show_pc12)

option_buttons_ui.append(show_gray_scale)


#Start augmentation
start_augmentation = st.button("🚀 Start augmentation")
#Start a new augmentation run
if start_augmentation and st.session_state.done:
    reset_for_new_run()


#----------------------------Sidebar for parameter---------------------------------
st.sidebar.header("⚙️ Parameter settings",
                    help="Change parameters of the selected augmentation method")

# General parameter for all augmentation techniques
st.sidebar.subheader("General")
AUG_COUNT = 5
#The number of augmentations generated per image
AUG_COUNT = st.sidebar.number_input("Number of augmentations", min_value=0, max_value=100,  value=getattr(constants, "AUG_COUNT", 3),
                                    help="The number of augmentations generated per image")

# Special color jitter parameter 
if aug_option == COLORJITTER_STR:
    st.sidebar.subheader("🧮 Color Jitter parameter")
    #Varies the image brightness
    BRIGHTNESS = st.sidebar.slider("Brightness", 0.0, 2.0, getattr(constants, "BRIGHTNESS", 0.5),
                step=0.05,
                help="Varies the image brightness. Values above 1 make the image brighter, values below 1 make it darker.")
    #Changes the contrast of the image
    CONTRAST = st.sidebar.slider("Contrast", 0.0, 2.0, getattr(constants, "CONTRAST", 0.5),
                step=0.05,
                help="Changes the contrast of the image. Higher values increase the difference between light and dark.")
    #Changes the color saturation
    SATURATION = st.sidebar.slider("Saturation", 0.0, 2.0, getattr(constants, "SATURATION", 0.5),
                step=0.05,   
                help="Changes the color saturation. Low values desaturate the image, high values intensify the colors.")
    #Shifts the color tone of the image
    HUE = st.sidebar.slider("Hue", 0.0, 0.5, getattr(constants, "HUE", 0.1),
                step=0.05,
                help="Shifts the color tone of the image. Small values result in subtle color shifts.")

    #Set values 
    constants.BRIGHTNESS = BRIGHTNESS
    constants.CONTRAST = CONTRAST
    constants.SATURATION = SATURATION
    constants.HUE = HUE
    
    
# Special fancy gng parameter 
elif aug_option == FANCYGNG_STR:
    st.sidebar.subheader("🧮 Fancy GNG parameter")
    #Determines the strength of the color shift along the PCA components
    STANDARD_DEVIATION = st.sidebar.slider("Standard deviation", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_STANDARD_DEVIATION", 2.0)),
                    step=0.05,
                    help="Determines the strength of the color shift along the PCA components. Higher values produce stronger color variations.")
    #Sets the average shift along the color PCA
    MEAN = st.sidebar.slider("Mean", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_MEAN", 3.0)),
                    step=0.05,
                    help="Sets the average shift along the color PCA. Affects how much colors are changed on average.")
    
    #Set values 
    constants.FANCY_PCA_STANDARD_DEVIATION = STANDARD_DEVIATION
    constants.FANCY_PCA_MEAN = MEAN
    
# Special fancy pca parameter 
elif aug_option == FANCYPCA_STR:
    st.sidebar.subheader("🧮 Fancy PCA parameter")
    #Determines the strength of the color shift along the PCA components
    STANDARD_DEVIATION = st.sidebar.slider("Standard deviation", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_STANDARD_DEVIATION", 2.0)),
                        step=0.05,
                        help="Determines the strength of the color shift along the PCA components. Higher values produce stronger color variations.")
    #Sets the average shift along the color PCA
    MEAN = st.sidebar.slider("Mean", 0.0, 10.0,  float(getattr(constants, "FANCY_PCA_MEAN", 3.0)),
                        step=0.05,
                        help="Sets the average shift along the color PCA. Affects how much colors are changed on average.")  
    
    #Set values 
    constants.FANCY_PCA_STANDARD_DEVIATION = STANDARD_DEVIATION
    constants.FANCY_PCA_MEAN = MEAN

# Special point cloud parameter 
if show_point_cloud:
    st.sidebar.subheader("☁️ Size of the point cloud")
    #Size of point cloud
    CLOUD_SIZE = st.sidebar.number_input("Number of points", 100, 1000000, CLOUD_SIZE,
                help="By default, 5000 random pixels from the image are selected for displaying the point cloud. " \
                "This helps to generate the point cloud faster and save computing power. " \
                "However, any other number smaller than the total number of pixels can also be selected. ")
    
    #By default the point cloud uses 5000 random data points in order to safe computational power. This option allows to use the whole image to generate the point cloud
    use_original_size = st.sidebar.checkbox("Use original image size",
                help="Use all pixels of the image to generate the point cloud. This may take some time for larger images.")

# Special PC1/PC2 parameter
if show_pc12:
    st.sidebar.subheader("📉 Size of the PC1/PC2 plot")
    #Size of PC1/PC2 plot
    PC12_SIZE = st.sidebar.number_input("Number of points", 100, 1000000, PC12_SIZE,
                help="By default, 5000 random pixels from the image are selected for displaying the PC1/PC2 plot. " \
                "This helps to generate the PC1/PC2 visualization faster and save computing power. " \
                "However, any other number smaller than the total number of pixels can also be selected. ")

    #By default the PC1/PC2 plot uses 5000 random data points in order to safe computational power. This option allows to use the whole image to generate the PC1/PC2 plot
    use_original_size_pc12 = st.sidebar.checkbox("Use original image size",
                help="Use all pixels of the image to generate the PC1/PC2 plot. This may take some time for larger images.")

# Special cluster cloud parameter   
if show_cluster_cloud:
    st.sidebar.subheader("☁️ Size of the cluster cloud")
    #Size of cluster cloud
    CLUSTER_CLOUD_SIZE = st.sidebar.number_input("Number of points", 100, 1000000, CLOUD_SIZE,
                help="By default, 5000 random pixels from the image are selected for displaying the cluster cloud. " \
                "This helps to generate the cluster cloud faster and save computing power. " \
                "However, any other number smaller than the total number of pixels can also be selected. ")
    
    #By default the cluster cloud uses 5000 random data points in order to safe computational power. This option allows to use the whole image to generate the cluster cloud
    use_original_size_cluster = st.sidebar.checkbox("Use original image size",
                help="Use all pixels of the image to generate the cluster cloud. This may take some time for larger images.")

# Special parmeter for fancy gng reduced training
if reduced_fancy_gng and aug_option == FANCYGNG_STR:
    st.sidebar.subheader("Number of pixels used for Fancy GNG training")
    #Number of data points used for the fancy gng training. 
    REDUCED_TRAINING = st.sidebar.number_input("Number of pixels", 100, 1000000, REDUCED_TRAINING,
                help="By default, the reduced GNG training uses 5000 random pixels from the image. " \
                "This helps to train GNG faster and save computing power. " \
                "However, any other number that is smaller than the total number of pixels can also be selected. ")

constants.AUG_COUNT = AUG_COUNT


#-----------------------------------FancyPCA------------------------------------------

#Basic fancy pca start function.
#Parameter: image_data -> 1D normalized image, original_image -> original rgb image
def fancy_pca(image_data, original_iamge):
    #generate fancy pca augmentations
    aug_images = generate_fancy_pca_augmentations(image_data, original_iamge.size)
    #save images in session
    st.session_state.image_results[filename] = {
                   "original": original_iamge,
                   "aug_images": aug_images,
                   "data_shape": image_data.shape,
    }

#Compute the fancy pca augmenation
#Parameter: image_data -> 1D normalized image, image_size -> original rgb image size
def generate_fancy_pca_augmentations(image_data, image_size):
    #create fancy pca instance
    fancy_pca_transform = FP.FancyPCA()
    #final augmentation array
    aug_images = []
    width, height = image_size
    channels = 3
    
    for _ in range(constants.AUG_COUNT):
        #Create fancy pca image
        fancy_pca_image = fancy_pca_transform.fancy_pca(image_data.copy())
        #Reshape image
        fancy_pca_image = (fancy_pca_image * 255).astype(np.uint8)
        fancy_pca_image = fancy_pca_image.reshape((height, width, channels))

        try:
            aug_image = Image.fromarray(fancy_pca_image)
            aug_images.append(aug_image)
        except Exception as e:
            st.write(f"Error during image conversion: {e}")  

    return aug_images

#Show fancy pca debug output in the ui
def show_fancy_pca_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Image size:** {info['original'].size}")
    st.write(f"**Image array shape:** {info['data_shape']}")





#-------------------------------------FancyGNG---------------------------------------------

#Basic fancy gng start function.
#Parameter: image_data -> 1D normalized image, original_image -> original rgb image
def fancy_gng(image_data, original_image):
    #generate fancy gng augmentations
    aug_images, cluster_count, pixel_cluster_map, node_cluster_map = generate_fancy_gng_augmentations(image_data, original_image.size)
    #save images in session
    st.session_state.image_results[filename] = {
                    "original": original_image,
                    "aug_images": aug_images,
                    "cluster_count": cluster_count,
                    "data_shape": image_data.shape,
                    "pixel_cluster_map": pixel_cluster_map,
                    "nodes": node_cluster_map.size
    }

#Show fancy gng debug output in the ui
def show_fancy_gng_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Image size:** {info['original'].size}")
    st.write(f"**Image array shape:** {info['data_shape']}")
    st.write(f"**Number of clusters:** {info['cluster_count']}")
    st.write(f"**Number of codebook vectors generated by GNG:** {info['nodes']}")

#Compute the fancy gng augmenation
#Parameter: image_data -> 1D normalized image, image_size -> original rgb image size
def generate_fancy_gng_augmentations(image_data, image_size):
     #create fancy gng instance
    gng = dbl_gng.DBL_GNG(3, constants.MAX_NODES)
    image_data_org = image_data.copy()

    #check for reduced training
    if reduced_fancy_gng and REDUCED_TRAINING < len(image_data):
        #sample random data points
        indices = np.random.choice(len(image_data), REDUCED_TRAINING, replace=False)
        image_data = image_data[indices]

    #init dbl nodes   
    gng.initializeDistributedNode(image_data, constants.STARTING_NODES)
    #progress bar
    bar = trange(constants.EPOCH)
    #start fancy gng training
    for i in bar:
        gng.resetBatch()
        gng.batchLearning(image_data)
        gng.updateNetwork()
        gng.addNewNode(gng)
        bar.set_description(f"Epoch {i + 1} number of nodes: {len(gng.W)}") 

    #Remove weak edges
    gng.cutEdge()
    gng.finalNodeDatumMap(image_data_org)

    finalDistMap = gng.finalDistMap
    
    #final fancy gng nodes
    finalNodes = (gng.W * constants.MAX_COLOR_VALUE).astype(int)
    connectiveMatrix = gng.C
    #create map: pixel -> cluster & node -> cluster
    pixel_cluster_map, node_cluster_map = clustering.cluster(finalDistMap, finalNodes, connectiveMatrix)
    pixel_cluster_map = np.array(pixel_cluster_map)
    #final amount of clusters
    cluster_count = int(max(node_cluster_map)) + 1

    width, height = image_size
    #final image augmentation array
    aug_images = []
    #Create pca augmentation based on the clusters
    for _ in range(constants.AUG_COUNT):
        #Fancy gng augmentation for each cluster
        aug_data = color_pca.modify_clusters(image_data_org, pixel_cluster_map, cluster_count, [image_size], 0)
        
        #reshape image
        aug_data = (aug_data * 255).astype(np.uint8) 
        aug_data = aug_data.reshape((height, width, 3))
        aug_image = Image.fromarray(aug_data)
        aug_images.append(aug_image)
    return aug_images, cluster_count, pixel_cluster_map, node_cluster_map



#-----------------------------------Color Jitter------------------------------------------

#Basic color jitter start function.
#Parameter: image_data -> 1D normalized image, original_image -> original rgb image
def color_jitter(image_data, original_image):
     #generate color jitter augmentations
    aug_images = generate_color_jitter_augmentations(original_image)
    #save images in session
    st.session_state.image_results[filename] = {
        "original": original_image,
        "aug_images": aug_images,
        "data_shape": image_data.shape,
        "parameter": {"Brightness" : constants.BRIGHTNESS,
                      "Contrast" : constants.CONTRAST,
                      "Saturation" : constants.SATURATION,
                      "Hue" : constants.HUE}
    }

#Compute the color jitter augmenation
#Parameter: image -> original image
def generate_color_jitter_augmentations(image):
    #create color jitter augmentation with the given parameters
    transform = transforms.ColorJitter(
        brightness=constants.BRIGHTNESS, contrast=constants.CONTRAST, saturation=constants.SATURATION, hue=constants.HUE
    )
    #final augmentation array
    aug_images = []
    for _ in range(constants.AUG_COUNT):
        img = transform(image)
        aug_images.append(img)
    return aug_images

#Show color jitter debug output in the ui
def show_color_jitter_info(filename, info):
    st.divider()
    st.subheader(f"📸 {filename}")
    st.write(f"**Image size:** {info['original'].size}")
    st.write(f"**Image array shape:** {info['data_shape']}")
    st.write(f"**Parameter:** {info['parameter']}")
#-----------------------------Plotting----------------------------------------------

#Compute shared point cloud data for normal point cloud and cluster cloud
def get_point_cloud_plot_data(all_images, cloud_size, use_original_cloud):
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    current_info = globals().get("info", {})
    cluster_source = current_info["pixel_cluster_map"] if "pixel_cluster_map" in current_info else None
    cluster_map = np.asarray(cluster_source) if cluster_source is not None else None
    cache_key = (
        tuple((id(img), img.size) for img in images),
        int(cloud_size),
        bool(use_original_cloud),
        id(cluster_source) if cluster_source is not None else None
    )
    cache = getattr(get_point_cloud_plot_data, "_cache", {})

    if cache_key in cache:
        return cache[cache_key]

    point_cloud_plot_data = []

    for img in images:
        rgb_image = img.convert("RGB")
        points = np.asarray(rgb_image).reshape(-1, 3)
        cluster_values = cluster_map if cluster_map is not None and len(cluster_map) == len(points) else None

        #Randomly select points (or all of them, if fewer)
        if len(points) > cloud_size and not use_original_cloud:
            indices = np.random.choice(len(points), cloud_size, replace=False)
            points = points[indices]
            if cluster_values is not None:
                cluster_values = cluster_values[indices]

        point_cloud_plot_data.append({
            "points": points,
            "cluster_values": cluster_values
        })

    cache[cache_key] = point_cloud_plot_data
    get_point_cloud_plot_data._cache = cache

    return point_cloud_plot_data


#Create point cloud figure 
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes , row_idx -> figure row of visualization
def create_point_cloud(all_images, axs, row_idx = 0):
    #Cap the figure images
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    point_cloud_plot_data = get_point_cloud_plot_data(all_images, CLOUD_SIZE, use_original_size)

    for idx, img in enumerate(images):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)

        #ui parameter of axis
        ax.tick_params(width=3, labelsize=TICK_LABEL_SIZE)
        
        # use only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0: 
            points = point_cloud_plot_data[idx]["points"]

            # Define points (r,g,b -> as color)
            ax.scatter(points[:, 1], points[:, 2], c=points[:, 0:3] / 255, s=3, alpha=1.0)
            #Set axis parameters
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_xticks(range(0, 256, 100))  
            ax.set_yticks(range(0, 256, 100)) 
            ax.set_aspect('equal', adjustable='box')
            ax.set_box_aspect(1)
            ax.set_title("Point Cloud (GB)" if idx == 0 else f"Point Cloud (GB) {idx}", fontsize=TITLE_FONT_SIZE, pad=PLOT_TITLE_PAD, y=PLOT_TITLE_Y)

#Compute PC1/PC2 basis from the original image
#Parameter: image -> original image
def compute_pc12_basis_from_original(image):
    rgb_image = image.convert("RGB")
    points = np.asarray(rgb_image).reshape(-1, 3).astype(np.float32) / constants.MAX_COLOR_VALUE

    mean_vec = np.mean(points, axis=0)
    mean_free_data = points - mean_vec
    data_covarianz = np.cov(mean_free_data, rowvar=False)
    eig_values, eig_vecs = np.linalg.eigh(data_covarianz)

    # sort descending -> PC1, PC2
    order = np.argsort(eig_values)[::-1]
    eig_vecs = eig_vecs[:, order]
    pc_basis = eig_vecs[:, :2]

    return mean_vec, pc_basis


#Compute shared PC1/PC2 projection data for normal PC1/PC2 and cluster PC1/PC2
def get_pc12_plot_data(all_images):
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    current_info = globals().get("info", {})
    cluster_source = current_info["pixel_cluster_map"] if "pixel_cluster_map" in current_info else None
    cluster_map = np.asarray(cluster_source) if cluster_source is not None else None
    cache_key = (
        tuple((id(img), img.size) for img in images),
        int(PC12_SIZE),
        bool(use_original_size_pc12),
        id(cluster_source) if cluster_source is not None else None
    )
    cache = getattr(get_pc12_plot_data, "_cache", {})

    if cache_key in cache:
        return cache[cache_key]

    if len(images) == 0:
        return {
            "projected_results": [],
            "global_x_min": None,
            "global_x_max": None,
            "global_y_min": None,
            "global_y_max": None,
            "x_pad": None,
            "y_pad": None,
            "error": None
        }

    try:
        # Use PCA basis only from the original image
        original_mean, pc_basis = compute_pc12_basis_from_original(images[0])

        projected_results = []
        global_x_min, global_x_max = np.inf, -np.inf
        global_y_min, global_y_max = np.inf, -np.inf

        # Project original + all augmentations into the SAME PCA basis
        for img in images:
            rgb_image = img.convert("RGB")
            points = np.asarray(rgb_image).reshape(-1, 3).astype(np.float32) / constants.MAX_COLOR_VALUE
            cluster_values = cluster_map if cluster_map is not None and len(cluster_map) == len(points) else None

            if len(points) > PC12_SIZE and not use_original_size_pc12:
                indices = np.random.choice(len(points), PC12_SIZE, replace=False)
                plot_points = points[indices]
                if cluster_values is not None:
                    cluster_values = cluster_values[indices]
            else:
                plot_points = points

            projected = np.dot(plot_points - original_mean, pc_basis)

            projected_results.append({
                "plot_points": plot_points,
                "cluster_values": cluster_values,
                "projected": projected
            })

            global_x_min = min(global_x_min, np.min(projected[:, 0]))
            global_x_max = max(global_x_max, np.max(projected[:, 0]))
            global_y_min = min(global_y_min, np.min(projected[:, 1]))
            global_y_max = max(global_y_max, np.max(projected[:, 1]))

        # Small padding so points do not touch the border
        x_pad = 0.05 * (global_x_max - global_x_min) if global_x_max > global_x_min else 0.01
        y_pad = 0.05 * (global_y_max - global_y_min) if global_y_max > global_y_min else 0.01

        pc12_plot_data = {
            "projected_results": projected_results,
            "global_x_min": global_x_min,
            "global_x_max": global_x_max,
            "global_y_min": global_y_min,
            "global_y_max": global_y_max,
            "x_pad": x_pad,
            "y_pad": y_pad,
            "error": None
        }

    except Exception as e:
        pc12_plot_data = {
            "projected_results": [],
            "global_x_min": None,
            "global_x_max": None,
            "global_y_min": None,
            "global_y_max": None,
            "x_pad": None,
            "y_pad": None,
            "error": e
        }

    cache[cache_key] = pc12_plot_data
    get_pc12_plot_data._cache = cache

    return pc12_plot_data


#Create PC1/PC2 figure 
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes , row_idx -> figure row of visualization
def create_pc12_plot(all_images, axs, row_idx = 0):
    #Cap the figure images
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]

    if len(images) == 0:
        return

    pc12_plot_data = get_pc12_plot_data(all_images)

    if pc12_plot_data["error"] is not None:
        for idx, _ in enumerate(images):
            ax = get_fig_ax(axs, row_idx, idx)
            ax.axis("off")
            ax.text(0.5, 0.5, f"PC1/PC2 error:\n{pc12_plot_data['error']}", ha="center", va="center", fontsize=12)
        return

    for idx, data in enumerate(pc12_plot_data["projected_results"]):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)

        #ui parameter of axis
        ax.tick_params(width=3, labelsize=TICK_LABEL_SIZE)

        # use only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0:
            plot_points = data["plot_points"]
            plot_scale = constants.MAX_COLOR_VALUE
            projected = data["projected"] * plot_scale
            pc_axis_min = (pc12_plot_data["global_x_min"] - pc12_plot_data["x_pad"]) * plot_scale
            pc_axis_max = (pc12_plot_data["global_x_max"] + pc12_plot_data["x_pad"]) * plot_scale
            pc_y_axis_min = (pc12_plot_data["global_y_min"] - pc12_plot_data["y_pad"]) * plot_scale
            pc_y_axis_max = (pc12_plot_data["global_y_max"] + pc12_plot_data["y_pad"]) * plot_scale
            ax.scatter(projected[:, 0], projected[:, 1], c=plot_points, s=3, alpha=1.0)
            ax.set_xlim(pc_axis_min, pc_axis_max)
            ax.set_ylim(pc_y_axis_min, pc_y_axis_max)
            ax.set_aspect('auto')
            ax.set_box_aspect(1)
            ax.set_xlim(pc_axis_min, pc_axis_max)
            ax.set_ylim(pc_y_axis_min, pc_y_axis_max)
            ax.set_title("PC1/PC2" if idx == 0 else f"PC1/PC2 {idx}", fontsize=TITLE_FONT_SIZE, pad=PLOT_TITLE_PAD, y=PLOT_TITLE_Y)


#Create point cluster figure 
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes , row_idx -> figure row of visualization
def create_cluster_cloud(all_images, axs, row_idx = 0):
    #Cap the figure images
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    cloud_size = CLOUD_SIZE if show_point_cloud else CLUSTER_CLOUD_SIZE
    use_original_cloud = use_original_size if show_point_cloud else use_original_size_cluster
    point_cloud_plot_data = get_point_cloud_plot_data(all_images, cloud_size, use_original_cloud)

    for idx, img in enumerate(images):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)
        #ui parameter of axis
        ax.tick_params(width=3, labelsize=TICK_LABEL_SIZE)
        
        # only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0:  
            data = point_cloud_plot_data[idx]
            points = data["points"]
            cluster_values = data["cluster_values"]
            
            #color points in cluster color
            if cluster_values is not None:
                colors = np.array(
                    [constants.get_color(int(c)) for c in cluster_values],dtype=np.float32
                ) / 255.0
            else:
                ax.axis("off")
                continue

            # Define points (r,g,b -> as color)
            ax.scatter(points[:, 1], points[:, 2], c=colors, s=3, alpha=1.0)
            #Set axis parameters
            ax.set_xlim(0, 255)
            ax.set_ylim(0, 255)
            ax.set_xticks(range(0, 256, 100))  
            ax.set_yticks(range(0, 256, 100)) 
            ax.set_aspect('equal', adjustable='box')
            ax.set_box_aspect(1)
            ax.set_title("Cluster Cloud" if idx == 0 else f"Cluster Cloud {idx}", fontsize=TITLE_FONT_SIZE, pad=PLOT_TITLE_PAD, y=PLOT_TITLE_Y)


#Create cluster PC1/PC2 figure
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes , row_idx -> figure row of visualization
def create_cluster_pc12_plot(all_images, axs, row_idx = 0):
    #Cap the figure images
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]

    if len(images) == 0:
        return

    pc12_plot_data = get_pc12_plot_data(all_images)

    if pc12_plot_data["error"] is not None:
        for idx, _ in enumerate(images):
            ax = get_fig_ax(axs, row_idx, idx)
            ax.axis("off")
            ax.text(0.5, 0.5, f"Cluster PC1/PC2 error:\n{pc12_plot_data['error']}", ha="center", va="center", fontsize=12)
        return

    for idx, data in enumerate(pc12_plot_data["projected_results"]):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)

        #ui parameter of axis
        ax.tick_params(width=3, labelsize=TICK_LABEL_SIZE)

        # use only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0:
            cluster_values = data["cluster_values"]
            if cluster_values is not None:
                plot_colors = np.array(
                    [constants.get_color(int(c)) for c in cluster_values], dtype=np.float32
                ) / 255.0
            else:
                ax.axis("off")
                continue

            plot_scale = constants.MAX_COLOR_VALUE
            projected = data["projected"] * plot_scale
            pc_axis_min = (pc12_plot_data["global_x_min"] - pc12_plot_data["x_pad"]) * plot_scale
            pc_axis_max = (pc12_plot_data["global_x_max"] + pc12_plot_data["x_pad"]) * plot_scale
            pc_y_axis_min = (pc12_plot_data["global_y_min"] - pc12_plot_data["y_pad"]) * plot_scale
            pc_y_axis_max = (pc12_plot_data["global_y_max"] + pc12_plot_data["y_pad"]) * plot_scale
            ax.scatter(projected[:, 0], projected[:, 1], c=plot_colors, s=3, alpha=1.0)
            ax.set_xlim(pc_axis_min, pc_axis_max)
            ax.set_ylim(pc_y_axis_min, pc_y_axis_max)
            ax.set_aspect('auto')
            ax.set_box_aspect(1)
            ax.set_xlim(pc_axis_min, pc_axis_max)
            ax.set_ylim(pc_y_axis_min, pc_y_axis_max)
            ax.set_title("Cluster PC1/PC2" if idx == 0 else f"Cluster PC1/PC2 {idx}", fontsize=TITLE_FONT_SIZE, pad=PLOT_TITLE_PAD, y=PLOT_TITLE_Y)


#Create cluster PC1/PC2 EV figure
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes , row_idx -> figure row of visualization
def create_cluster_pc12_eigenvectors_plot(all_images, axs, row_idx = 0):
    #Cap the figure images
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]

    if len(images) == 0:
        return

    pc12_plot_data = get_pc12_plot_data(all_images)

    if pc12_plot_data["error"] is not None:
        for idx, _ in enumerate(images):
            ax = get_fig_ax(axs, row_idx, idx)
            ax.axis("off")
            ax.text(0.5, 0.5, f"Cluster PC1/PC2 EV error:\n{pc12_plot_data['error']}", ha="center", va="center", fontsize=12)
        return

    for idx, data in enumerate(pc12_plot_data["projected_results"]):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)

        #ui parameter of axis
        ax.tick_params(width=3, labelsize=TICK_LABEL_SIZE)

        # use only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0:
            cluster_values = data["cluster_values"]
            if cluster_values is not None:
                plot_colors = np.array(
                    [constants.get_color(int(c)) for c in cluster_values], dtype=np.float32
                ) / 255.0
            else:
                ax.axis("off")
                continue

            projected = data["projected"]
            ax.scatter(projected[:, 0], projected[:, 1], c=plot_colors, s=3, alpha=1.0)
            
            # Additional EV logic
            unique_clusters = np.unique(cluster_values)
            for c_id in unique_clusters:
                c_mask = (cluster_values == c_id)
                c_points = projected[c_mask]
                
                if len(c_points) > 1:
                    mean = np.mean(c_points, axis=0)
                    cov = np.cov(c_points, rowvar=False)
                    
                    try:
                        eigvals, eigvecs = np.linalg.eigh(cov)
                        
                        # Sort eigenvectors by eigenvalues in descending order
                        order = np.argsort(eigvals)[::-1]
                        eigvals = eigvals[order]
                        eigvecs = eigvecs[:, order]
                        
                        for i in range(2):
                            # Ensure we don't sqrt negative values from numerical imprecision
                            if eigvals[i] > 0:
                                ev_scale = np.sqrt(eigvals[i]) * 1.0
                                
                                # Plot quiver originating from mean
                                ax.quiver(mean[0], mean[1], 
                                          eigvecs[0, i] * ev_scale, eigvecs[1, i] * ev_scale, 
                                          angles='xy', scale_units='xy', scale=1, 
                                          color='black', width=0.007, zorder=3)
                    except np.linalg.LinAlgError:
                        pass # Ignore if decomposition fails for degenerate clusters

            ax.set_xlim(pc12_plot_data["global_x_min"] - pc12_plot_data["x_pad"], pc12_plot_data["global_x_max"] + pc12_plot_data["x_pad"])
            ax.set_ylim(pc12_plot_data["global_x_min"] - pc12_plot_data["x_pad"], pc12_plot_data["global_x_max"] + pc12_plot_data["x_pad"])
            ax.set_aspect('auto')
            ax.set_box_aspect(1)
            ax.set_title("Cluster PC1/PC2 EV" if idx == 0 else f"Cluster PC1/PC2 EV {idx}", fontsize=TITLE_FONT_SIZE, pad=PLOT_TITLE_PAD, y=PLOT_TITLE_Y)


#Create gray image figure 
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes , row_idx -> figure row of visualization
def create_gray_images(all_images, axs, row_idx = 0):
    #Cap the figure images
    images = all_images if len(all_images) <= MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]
    #Create gray scale transform
    grayscale_transform = transforms.Grayscale()
    st.session_state.gray_images[filename] = {"images": []}

    for idx, img in enumerate(images):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)

        # only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0:
            #create gray scale image
            gray = grayscale_transform(img)
            if idx != 0:
                #save gray scale image in session
                st.session_state.gray_images[filename]["images"].append(gray)
            if figures:
                #show image 
                ax.imshow(gray, cmap="gray")
                ax.axis("off")
                ax.set_box_aspect(1)
                ax.set_anchor('N')
                ax.set_title("Grayscale" if idx == 0 else f"Grayscale {idx}", fontsize=TITLE_FONT_SIZE, pad=IMAGE_TITLE_PAD, y=IMAGE_TITLE_Y)

    #Generate the remaining gray images (which ar not shown in the figure)
    if len(all_images) > MAX_UI_AUG_COUNT:
        for img in all_images[MAX_UI_AUG_COUNT:]:
            gray = grayscale_transform(img)
            #save gray scale image in session
            st.session_state.gray_images[filename]["images"].append(gray)


#Create pixel cluster figure 
#Parameter: all_images -> Array of original image & augmentated images, cluster_ax -> special side figure axis
def create_cluster_image(all_images, cluster_ax):
    # original image
    img = all_images[0]

    # only at empty axes
    if len(cluster_ax.images) == 0 and len(cluster_ax.collections) == 0:
        width, height = img.size
        #create new image with original image size
        image = Image.new("RGB", (width, height))
        draw = ImageDraw.Draw(image)

        tmp_width, tmp_height = 0, 0
        cluster = info['pixel_cluster_map']

        #color every pixel in the cluster color
        for group in cluster:
            #get cluster color
            color = constants.get_color(int(group))
            draw.point((tmp_width, tmp_height), fill=color)

            tmp_width += 1
            if tmp_width >= width:
                tmp_width = 0
                tmp_height += 1

        #show final image
        cluster_ax.imshow(image)
        cluster_ax.axis("off")
        cluster_ax.set_box_aspect(1)
        cluster_ax.set_anchor('N')
        cluster_ax.set_title("Cluster Map", fontsize=TITLE_FONT_SIZE, pad=IMAGE_TITLE_PAD, y=IMAGE_TITLE_Y)


#Create original image & augmentation images figure 
#Parameter: all_images -> Array of original image & augmentated images, axs -> fig axes, row_idx -> figure row of visualization
def create_main_plot(all_images, axs, row_idx = 0):

    #Cap the figure images
    images = all_images if len(all_images) < MAX_UI_AUG_COUNT else all_images[:MAX_UI_AUG_COUNT]

    for idx, img in enumerate(images):
        #get current axis
        ax = get_fig_ax(axs, row_idx, idx)
        # only at empty axes
        if len(ax.images) == 0 and len(ax.collections) == 0: 
            #show image 
            ax.imshow(img)
            ax.axis("off")
            ax.set_box_aspect(1)
            ax.set_anchor('N')
            ax.set_title("Original" if idx == 0 else f"Aug {idx}", fontsize=TITLE_FONT_SIZE, pad=IMAGE_TITLE_PAD, y=IMAGE_TITLE_Y)


#Create a png out of the figure
#Parameter: fig -> The created figure
def fig_to_png(fig):
    #create buffer for figure png
    buf = io.BytesIO()
    #save fig as png
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    return buf






#---------------------------Helper functions-------------------------------------------------

#def keep_dependent_ui_element_at_random_button(dependency, dependency_func_map : dict):
#    if dependency is not None: 
#        entry = dependency_func_map.get(dependency)
#        if entry and len(entry) > 0:
#            func = entry[0]
#            args = entry[1:]
#            return func(*args)  


#Get the current axis based on the row
#Parameter: axs -> fig axes, row_idx -> current row, idx -> image counter
def get_fig_ax(axs, row_index, idx):
    if axs.ndim == 1:
        return axs[idx]
    else:
        return axs[row_index, idx]








#------------------------------------------Main processing----------------------------------------------------------------------------------

#Start check for augmentation
if (start_augmentation or st.session_state.done) and st.session_state.uploaded_files:
    #iterate over each uploaded image
    for uploaded_file in st.session_state.uploaded_files:
        #save file name
        filename = uploaded_file.name
        # If already calculated, skip computation
        if filename not in st.session_state.image_results and start_augmentation:
            #save augmentation technique in session
            st.session_state.last_aug = aug_option
            with st.spinner(f"Process {filename} ..."):
                image = Image.open(uploaded_file).convert("RGB")
                
                
                # Resize image if it exceeds 640 width or 480 height
                if image.width > 640 or image.height > 480:
                    image.thumbnail((640, 480), Image.Resampling.LANCZOS)
                
                #image as array
                image_array = np.asarray(image)
                data_array = image_array.reshape(-1, 3) / constants.MAX_COLOR_VALUE

                #call fancy gng
                if aug_option == FANCYGNG_STR:
                    fancy_gng(data_array, image)
                
                #call fancy pca
                elif aug_option == FANCYPCA_STR:
                    fancy_pca(data_array, image)

                #call color jitter
                elif aug_option == COLORJITTER_STR:
                    color_jitter(data_array, image)
                

                
                
    
        # Get augmentation session data structure
        info = st.session_state.image_results[filename]

        #save augmentation debug info 
        if figures:
            if filename not in st.session_state.fig_png:
                if aug_option == FANCYGNG_STR:
                    st.session_state.last_aug_info = show_fancy_gng_info
           
                elif aug_option == FANCYPCA_STR:
                    st.session_state.last_aug_info = show_fancy_pca_info

                elif aug_option == COLORJITTER_STR:
                    st.session_state.last_aug_info = show_color_jitter_info
                
            
            #show augmentation debug info 
            st.session_state.last_aug_info(filename, info)
        

        #Create the figure for visualization
        with st.spinner(f"Augmentation of {filename} completed ... Start visualization"):
            #If no figure is created yet
            if filename not in st.session_state.fig_png:
                start_time = time.time()
                #get current image
                image = info["original"]
                #get figure rows 
                rows = sum(1 for opt in option_buttons_ui if opt and opt) + 1
                #get figure columns
                cols = constants.AUG_COUNT + 1 if constants.AUG_COUNT < MAX_UI_AUG_COUNT else MAX_UI_AUG_COUNT
                #create basic figure
                extra_col = 1 if show_cluster and figures else 0
                total_cols = cols + extra_col
                fig = plt.figure(figsize=(5.5 * total_cols, 5.5 * rows), dpi=100)
                #create figure axis
                axs = np.empty((rows, cols), dtype=object)

                #Create special side column for cluster image
                extra_cluster_axes = []
                if show_cluster and figures:
                    gs = fig.add_gridspec(rows, cols + 1, width_ratios=[1] * (cols + 1))
                    cluster_ax = fig.add_subplot(gs[0, -1])
                    for rr in range(1, rows):
                        empty_ax = fig.add_subplot(gs[rr, -1])
                        empty_ax.axis("off")
                        extra_cluster_axes.append(empty_ax)
                else:
                    gs = fig.add_gridspec(rows, cols)
                #iterate over each row and column
                for r in range(rows):
                    for c in range(cols):
                        #add sub plots to axis               
                        axs[r, c] = fig.add_subplot(gs[r, c])
               

                current_row = 0
                #Generate figure row with the original image & the augmentated images
                if figures:
                    create_main_plot([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1
                    
                # Generate figure row with the point clouds 
                if show_point_cloud and figures:
                    create_point_cloud([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                # Generate figure row with the PC1/PC2 plots
                if show_pc12 and figures:
                    create_pc12_plot([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                # Generate figure row with the cluster clouds
                if show_cluster_cloud and figures:
                    create_cluster_cloud([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                # Generate figure row with the cluster PC1/PC2 plots
                if show_cluster_cloud and show_pc12 and figures:
                    create_cluster_pc12_plot([info["original"]] + info["aug_images"], axs, current_row)
                    current_row += 1

                # Generate figure row with the gray scale images
                if show_gray_scale:
                    create_gray_images([info["original"]] + info["aug_images"], axs, current_row)
                
                # Generate figure column with the cluster image
                if show_cluster and figures:
                    create_cluster_image([info["original"]] + info["aug_images"], cluster_ax)
                
                #save figure as png
                if figures:   
                    fig.subplots_adjust(wspace=0.35, hspace=0.55)   
                    png_buf = fig_to_png(fig)
                    st.session_state.fig_png[filename] = png_buf.getvalue()
                    plt.close(fig)
                    
                    end_time = time.time()
                    st.success(f"Visualization generated in {end_time - start_time:.2f} seconds!")
               
                    
            
        #Show final figure/png
        if figures:
            st.image(st.session_state.fig_png[filename])
    st.session_state.done = True
        




#--------------------------------------------Download----------------------------------------------

#Check if there is an augmentation result
if st.session_state.image_results:
    st.divider()
    #create zip buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        #save each augmented image in zip file
        for filename, info in st.session_state.image_results.items():
            base_name = filename.rsplit('.', 1)[0]
            for i, img in enumerate(info["aug_images"]):
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                #create file & file name
                zipf.writestr(f"{base_name}_aug_{st.session_state.last_aug}_{i+1}.jpg", buf.getvalue())
    
    #Augmentation download button
    download = st.download_button(
        label="⬇️ Download augmented images as a ZIP file",
        data=zip_buffer.getvalue(),
        file_name="augmented_images.zip",
        mime="application/zip",
    )


#Check if there is an gray scale result   
if st.session_state.image_results and filename in st.session_state.gray_images:
    #create zip buffer
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zipf:
        #save each augmented image in zip file
        for filename, info in st.session_state.gray_images.items():
            base_name = filename.rsplit('.', 1)[0]
            for i, img in enumerate(info['images']):
                buf = io.BytesIO()
                img.save(buf, format="JPEG")
                #create file & file name
                zipf.writestr(f"{base_name}_gray_scale_{st.session_state.last_aug}_{i+1}.jpg", buf.getvalue())
    
    #Gray scale download button
    download = st.download_button(
        label="⬇️ Download grayscale images as a ZIP file",
        data=zip_buffer.getvalue(),
        file_name="gray_scale_images.zip",
        mime="application/zip",
    )