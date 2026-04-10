import rasterio, warnings
import cv2, os
import numpy as np
import geopandas as gpd
from tqdm import tqdm


def load_stacked_tiff(file_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=rasterio.errors.NotGeoreferencedWarning)
        with rasterio.open(file_path) as src:
            return src.read(), src.transform, src.crs

def warpImages(img1, img2, H):
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2) #coordinates of a reference image
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2) #coordinates of second image

    # When we have established a homography we need to warp perspective
    # Change field of view
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)#calculate the transformation matrix

    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)
    
    translation_dist = [-x_min,-y_min]
    
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    output_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min))
    output_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    return output_img


def save_tif(save_path, final_result, final_transform, final_crs, aoi_id):
    output_file = os.path.join(save_path, f"aoi_{aoi_id}_stitched.tif")
    
    num_bands, h, w = final_result.shape
    
    new_profile = {
        'driver': 'GTiff',
        'dtype': final_result.dtype,
        'nodata': 0,
        'width': w,
        'height': h,
        'count': num_bands,
        'crs': final_crs,
        'transform': final_transform,
        'compress': 'lzw'
    }

    with rasterio.open(output_file, 'w', **new_profile) as dst:
        dst.write(final_result)
    print(f"Saved: {output_file}")


if __name__ == "__main__":
    root_folder = "091425_Wallpe"
    for aoi_id in tqdm(range(0, 24), desc="Processing AOIs"):
        rgb_path = os.path.join(root_folder, "Processed/thumbnails")
        aoi_geojson = os.path.join(root_folder, f"Processed/AOI/aoi_{aoi_id}.geojson")
        save_path = os.path.join(root_folder, f"Processed/stitched/aoi_{aoi_id}")
        os.makedirs(save_path, exist_ok=True)

        gdf = gpd.read_file(aoi_geojson)
        img_list = [cv2.imread(os.path.join(rgb_path, f"{name}.jpg")) for name in gdf['image_name']]

        sift = cv2.SIFT_create()
        idx = 1
        while True:
            img1 = img_list.pop(0)
            img2 = img_list.pop(0)

            kp_a, des_a = sift.detectAndCompute(img1, None)
            kp_t, des_t = sift.detectAndCompute(img2, None)

            bf = cv2.BFMatcher(cv2.NORM_L2)
            matches = bf.knnMatch(des_a, des_t, k=2)
            good = [m for m, n in matches if m.distance < 0.75 * n.distance]


            img3 = cv2.drawMatches(img1, kp_a, img2, kp_t, good, None, flags=2)
            cv2.imwrite(os.path.join(save_path, f"matches_{idx}.png"), img3)
            idx += 1


            src_pts = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_t[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)  
            H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            result = warpImages(img2, img1, H)
            img_list.insert(0,result)

            if len(img_list) == 1:
                cv2.imwrite(os.path.join(save_path, f"final_stitched_aoi_{aoi_id}.png"), result)
                break
