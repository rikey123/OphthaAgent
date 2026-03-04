import os
import subprocess
import pandas as pd
import shutil

def quantitative_measurement(image_path):
    """
    Performs quantitative measurement of fundus images by running AutoMorph tools.

    Args:
        image_path (str): The relative path to the input image.

    Returns:
        dict: A dictionary containing the measurement results.
    """
    automorph_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'AutoMorph', 'Results'))
    
    # Define input directories for the AutoMorph tools
    disc_centred_artery_skeleton_dir = os.path.join(automorph_data_dir, 'M2', 'artery_vein', 'disc_centred_artery_skeleton')
    disc_centred_vein_skeleton_dir = os.path.join(automorph_data_dir, 'M2', 'artery_vein', 'disc_centred_vein_skeleton')
    disc_centred_binary_skeleton_dir = os.path.join(automorph_data_dir, 'M2', 'binary_vessel', 'disc_centred_binary_skeleton')
    zone_b_disc_centred_artery_skeleton_dir = os.path.join(automorph_data_dir, 'M2', 'artery_vein', 'Zone_B_disc_centred_artery_skeleton')
    zone_b_disc_centred_vein_skeleton_dir = os.path.join(automorph_data_dir, 'M2', 'artery_vein', 'Zone_B_disc_centred_vein_skeleton')
    zone_b_disc_centred_binary_skeleton_dir = os.path.join(automorph_data_dir, 'M2', 'binary_vessel', 'Zone_B_disc_centred_binary_skeleton')

    # Create the directories if they don't exist
    os.makedirs(disc_centred_artery_skeleton_dir, exist_ok=True)
    os.makedirs(disc_centred_vein_skeleton_dir, exist_ok=True)
    os.makedirs(disc_centred_binary_skeleton_dir, exist_ok=True)
    os.makedirs(zone_b_disc_centred_artery_skeleton_dir, exist_ok=True)
    os.makedirs(zone_b_disc_centred_vein_skeleton_dir, exist_ok=True)
    os.makedirs(zone_b_disc_centred_binary_skeleton_dir, exist_ok=True)

    # Copy the input image to the appropriate directories
    image_name = os.path.basename(image_path)
    shutil.copy(image_path, os.path.join(disc_centred_artery_skeleton_dir, image_name))
    shutil.copy(image_path, os.path.join(disc_centred_vein_skeleton_dir, image_name))
    shutil.copy(image_path, os.path.join(disc_centred_binary_skeleton_dir, image_name))
    shutil.copy(image_path, os.path.join(zone_b_disc_centred_artery_skeleton_dir, image_name))
    shutil.copy(image_path, os.path.join(zone_b_disc_centred_vein_skeleton_dir, image_name))
    shutil.copy(image_path, os.path.join(zone_b_disc_centred_binary_skeleton_dir, image_name))

    # Run the disc/cup segmentation script
    disc_cup_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'AutoMorph', 'M2_lwnet_disc_cup', 'generate_av_results.py'))
    subprocess.run(['python', disc_cup_script_path, '--config_file', 'experiments/wnet_All_three_1024_disc_cup/30/config.cfg', '--im_size', '512', '--device', 'cuda:0', '--image_path', image_path], check=True)

    # Run the feature measurement scripts
    whole_pic_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'AutoMorph', 'M3_feature_whole_pic', 'retipy', 'create_datasets_disc_centred.py'))
    zone_script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'AutoMorph', 'M3_feature_zone', 'retipy', 'create_datasets_disc_centred_B.py'))
    
    subprocess.run(['python', whole_pic_script_path], check=True)
    subprocess.run(['python', zone_script_path], check=True)

    # Read the results from the CSV files
    disc_measurement_path = os.path.join(automorph_data_dir, 'M3', 'Disc_centred', 'Disc_Measurement.csv')
    disc_zone_b_measurement_path = os.path.join(automorph_data_dir, 'M3', 'Disc_centred', 'Disc_Zone_B_Measurement.csv')

    results = {}
    if os.path.exists(disc_measurement_path):
        results['disc_measurement'] = pd.read_csv(disc_measurement_path).to_dict()
    if os.path.exists(disc_zone_b_measurement_path):
        results['disc_zone_b_measurement'] = pd.read_csv(disc_zone_b_measurement_path).to_dict()

    # Clean up the copied images
    os.remove(os.path.join(disc_centred_artery_skeleton_dir, image_name))
    os.remove(os.path.join(disc_centred_vein_skeleton_dir, image_name))
    os.remove(os.path.join(disc_centred_binary_skeleton_dir, image_name))
    os.remove(os.path.join(zone_b_disc_centred_artery_skeleton_dir, image_name))
    os.remove(os.path.join(zone_b_disc_centred_vein_skeleton_dir, image_name))
    os.remove(os.path.join(zone_b_disc_centred_binary_skeleton_dir, image_name))

    return results

if __name__ == '__main__':
    # Create a dummy image for testing
    if not os.path.exists('test_image.png'):
        from PIL import Image
        img = Image.new('RGB', (100, 100), color = 'red')
        img.save('test_image.png')

    results = quantitative_measurement('test_image.png')
    print(results)