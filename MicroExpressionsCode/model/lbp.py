from skimage.filters import gabor_kernel
from skimage import io
from scipy import ndimage as ndi

from model.utils import *
from model.svm import *
from main_dir.consts import *



# endregion

def get_lgbp_histogram(img):
    # convert image to grey
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # in the LGBPHS paper, they resized the photos to 80X88
    img = cv2.resize(img, (80, 88), interpolation=cv2.INTER_CUBIC)

    # create gabor filters
    filters = create_gabor_filter()

    # apply the filters on the image, in the LGBPHS paper, total of 40 maps
    images = apply_filters(img, filters)

    # create lbp map
    images = get_lbp_images(images)

    # split to regions, in th LGBPHS paper, total of m regions (each region 4x8) we will use 20x10, total of 200 regions
    images = split_images_to_regions(images)

    # create histograms
    lgbp_histogram = get_histograms_from_images(images, show=True)

    return lgbp_histogram


def split_images_to_regions(images, width=4, height=8, show=False):
    # print("[INFO] splitting images to regions...")
    images_tiles = []
    for image in tqdm(images, desc="[INFO] splitting images to regions..."):
        for y in range(0, image.shape[1], height):
            for x in range(0, image.shape[0], width):
                images_tiles.append(image[x:x + width, y:y + height])

    if show:
        show_grid(images_tiles, 80, 50)

    return images_tiles



def get_lbp_histograms(faces):
    # print("[INFO] Getting lbp maps...")
    lbp_histograms = []
    for image in tqdm(faces, desc="[INFO] Getting lbp maps..."):
        lbp_image = get_lbp_image(image)
        lbp_histogram = get_histogram_from_image(lbp_image)
        lbp_histograms.append(lbp_histogram)

    lbp_histograms = np.array(lbp_histograms)

    return lbp_histograms


def get_lbp_images(images, show=False):
    # print("[INFO] Getting lbp maps...")
    lbp_images = []
    for image in tqdm(images, desc="[INFO] Getting lbp maps..."):
        lbp_images.append(get_lbp_image(image))

    if show:
        show_grid(lbp_images)

    return lbp_images


def get_lbp_image(image):
    '''
    == Input ==
    gray_image  : color image of shape (height, width)
    == Output ==
    imgLBP : LBP converted image of the same shape as
    '''

    # Step 0: Step 0: Convert an image to grayscale
    img_lbp = np.zeros_like(image)
    neighboor = 3

    for ih in range(0, image.shape[0] - neighboor):
        for iw in range(0, image.shape[1] - neighboor):
            # Step 1: 3 by 3 pixel
            img = image[ih:ih + neighboor, iw:iw + neighboor]
            center = img[1, 1]
            img01 = (img >= center) * 1.0
            img01_vector = img01.T.flatten()

            # it is ok to order counterclock manner
            # img01_vector = img01.flatten()
            # Step 2: **Binary operation**:
            img01_vector = np.delete(img01_vector, 4)

            # Step 3: Decimal: Convert the binary operated values to a digit. 
            where_img01_vector = np.where(img01_vector)[0]
            if len(where_img01_vector) >= 1:
                num = np.sum(2 ** where_img01_vector)
            else:
                num = 0
            img_lbp[ih + 1, iw + 1] = num
    return (img_lbp)


def get_histogram_from_image(img):
    img_histogram = cv2.calcHist([img], [0], None, [256], [0, 256])
    return img_histogram


def get_histograms_from_images(images, show=False):
    # print("[INFO] Getting histograms...")
    histograms = []
    for image in tqdm(images, desc="[INFO] Getting histograms..."):
        histograms.append(get_histogram_from_image(image))

    histograms = np.asarray(histograms)
    histograms = histograms.flatten()
    return histograms



def create_gabor_filter():
    # This function is designed to produce a set of GaborFilters
    # an even distribution of theta values equally distributed amongst pi rad / 180 degree
    print("[INFO] Getting gabor filters...")
    filters = []

    scales = range(1, 6)  # lambda
    orientations = range(0, 8)  # theta

    for scale in scales:
        for orientation in orientations:
            kernel = np.real(gabor_kernel(frequency=scale, theta=orientation))
            filters.append(kernel)
    return filters


def apply_filters(img, filters, show=False):
    # This general function is designed to apply filters to our image
    # print("[INFO] applying filters on image...")

    # save new images
    gabor_images = []

    for kernel in tqdm(filters, desc="[INFO] applying filters on image..."):  # Loop through the kernels in our GaborFilter
        image_filter = ndi.convolve(img, kernel, mode="wrap")  # Apply filter to image
        gabor_images.append(image_filter)

    if show:
        show_grid(gabor_images)

    return gabor_images


def show_grid(images, ncols=8, nrows=5):
    # show images
    # grid_images = [cv2.resize(image, (100,100)) for image in gabor_images]
    grid_images = [np.concatenate(images[i*ncols: (i+1)*ncols], axis=1) for i in range(nrows)]
    grid_images = np.concatenate(grid_images, axis=0)
    cv2.imshow("grid_images", grid_images)
    cv2.waitKey(0)
    cv2.destroyAllWindows()