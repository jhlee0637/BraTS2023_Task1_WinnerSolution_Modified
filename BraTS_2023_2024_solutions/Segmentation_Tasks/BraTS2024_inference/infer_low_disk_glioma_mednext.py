print("START")

import os
from os.path import join 
import shutil

import multiprocessing
import concurrent.futures
from multiprocessing.pool import Pool
from concurrent.futures import ThreadPoolExecutor
import subprocess

import SimpleITK as sitk
import numpy as np
from scipy.ndimage import label 
import nibabel as nib
import cc3d
import scipy
import pandas as pd
import math

from batchgenerators.utilities.file_and_folder_operations import *
#import surface_distance

################################################################################
########### Convert the folder from BraTS to nnUNet ############################
################################################################################

def copy_BraTS_segmentation_and_convert_labels_from_BraTS(in_file, out_file):
    """
    Converts the labels from BraTS2024 convention to the nnUNet
    # IN:
        in_file: Path for the file from a BraTS folder
        out_file: Path for the new file
    # OUT:
        NONE
    """
    #The input of the nnUNet requires labels to be sequencial, so the ED — label 2 was replaced with label 1 and NCR — label 1 with label 2 for training 
    img = sitk.ReadImage(in_file)
    img_npy = sitk.GetArrayFromImage(img)

    uniques = np.unique(img_npy)
    for u in uniques:
        if u not in [0, 1, 2, 3, 4]:
            raise RuntimeError('unexpected label')

    seg_new = np.zeros_like(img_npy)
    seg_new[img_npy == 4] = 4
    seg_new[img_npy == 3] = 3
    seg_new[img_npy == 2] = 1
    seg_new[img_npy == 1] = 2
    img_corr = sitk.GetImageFromArray(seg_new)
    img_corr.CopyInformation(img)
    sitk.WriteImage(img_corr, out_file)

def convert_case_to_nnUNet(case_dir, imagestr):
    """
    Converts a folder from BraTS2024 to nnUNet convention
    #IN:
        case_dir: Path to a folder of BraTS2024 with all 4 modalities 
                    (For testing, labels are not present).
        imagestr: Path to save the converted files for the nnUNet
    #OUT:
        NONE
    """
    c = os.path.basename(case_dir)
    shutil.copy(os.path.join(case_dir, c + "-t1c.nii.gz"), os.path.join(imagestr, c + '_0000.nii.gz')) # t1ce -> t1c
    shutil.copy(os.path.join(case_dir, c + "-t1n.nii.gz"), os.path.join(imagestr, c + '_0001.nii.gz')) # t1 -> t1n
    shutil.copy(os.path.join(case_dir, c + "-t2f.nii.gz"), os.path.join(imagestr, c + '_0002.nii.gz')) # flair -> t2f
    shutil.copy(os.path.join(case_dir, c + "-t2w.nii.gz"), os.path.join(imagestr, c + '_0003.nii.gz')) # t2 -> t2w
    #copy_BraTS_segmentation_and_convert_labels_from_BraTS(os.path.join(case_dir, c + "-seg.nii.gz"), os.path.join(labelstr, c + '.nii.gz'))

def convert_dataset_to_nnUNet_parallel(dataset_data_dir, imagestr):
    """
    Making conversion from BraTS2024 to nnUNet parallel 
    #IN:
        dataset_data_dir: root path where folders for each case are.
        imagestr: path to save the converted files
    #OUT:
        NONE
    """
    case_ids = subdirs(dataset_data_dir, prefix='BraTS', join=False)
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        for c in case_ids:
            executor.submit(convert_case_to_nnUNet, os.path.join(dataset_data_dir, c), imagestr)

def convert_data_step(input_folder_nnunet, raw_dataset):
    """
    Function that calls the convertion function
    #IN:
        input_folder_nnunet: Path to savve the converted files. 
                                It is the folder where the nnUNet will get the files from.
        raw_dataset: Path where the organizers placed the testing dataset. 
                        It will be delivered by the argument "data_path"
    #OUT:
        NONE
    """
    print(f"Creating the {input_folder_nnunet}")
    maybe_mkdir_p(input_folder_nnunet) # Make the folder to save the converted files
    # Convert the dataset to nnUNet format in parallel
    convert_dataset_to_nnUNet_parallel(dataset_data_dir=raw_dataset,
                                        imagestr=input_folder_nnunet)


################################################################################
# Performe inference for each model
################################################################################

def correct_shape_prob(original_file_path, pkl_file, prob):
    # Create a sample 3D array (for example, a shape of (5, 5, 5))
    nifti_img = nib.load(original_file_path)
    original_array = nifti_img.get_fdata()

    with open(pkl_file, 'rb') as file:
        data = pickle.load(file)
        # (start_x, end_x, start_y, end_y, start_z, end_z)
        crop_box = data['crop_bbox']  # (start_x, end_x, start_y, end_y, start_z, end_z)

    # Calculate the required padding
    pad_width_x = (crop_box[0][0], original_array.shape[2] - crop_box[0][1])  # (before, after) for x 28, 30
    pad_width_y = (crop_box[1][0], original_array.shape[1] - crop_box[1][1])  # (before, after) for y
    pad_width_z = (crop_box[2][0], original_array.shape[0] - crop_box[2][1])  # (before, after) for z 0, 

    # Create a tuple of padding widths
    pad_widths = ((0,0), pad_width_x, pad_width_y, pad_width_z)

    # Pad the original array
    padded_array = np.pad(prob, pad_widths, mode='constant', constant_values=0)
    
    padded_array = padded_array[1:, :, :, :]
   
    return padded_array


def ensemble_predictions(inference_folder, ensemble_prob_path, number_models=3, first_infer=False):
    """
    Performs ensemble step by step, i.e., 
    after each inference it divides the prob maps by the number of models and sum with the existent prob maps,
    if they do not exist yet, creates new ones.
    #IN:
        inference_folder: Folder containing the prob maps after inference
        ensemble_prob_path: Final folder path where the prob maps will be saved after ensemble 
        number_models: number of models that will be sum to each prob map
        first_infer: if true created new npz files with new prob
    """
    print(f"inference_folder: {inference_folder}")
    list_of_files = listdir(inference_folder)
    assert len(list_of_files), 'At least one file must be given in list_of_files'
    prob_map = None
    for f in list_of_files:
        print(f)
        if f.endswith("nii.gz"):
            npz_name = f"{f.split('.nii.gz')[0]}.npz"
            file_path_nii = join(inference_folder, f) #Path to each inference
            file_path_npz = join(inference_folder, npz_name)
            pred_npz = np.load(file_path_npz) #Load inference 
            if 'probabilities' in pred_npz:
                prob_map = pred_npz['probabilities'] # Load probability maps
            else:
                prob_map = pred_npz['softmax'] # Load probability maps
                original_file_path = join(inference_folder, f"{f}")
                pkl_file = join(inference_folder, f"{f.split('.nii.gz')[0]}.pkl")
                prob_map = correct_shape_prob(original_file_path=original_file_path, pkl_file=pkl_file, prob=prob_map)
            
            # maybe increase precision to prevent rounding errors
            if prob_map.dtype != np.float32:
                prob_map = prob_map.astype(np.float32)
            # Divide by the number total of models
            prob_map /= number_models 
            #pred_nii = nib.load(file_path_nii)
            if first_infer:
                #target_image_with_header = nib.Nifti1Image(prob_map, pred_nii.affine, header=pred_nii.header)
                #nib.save(target_image_with_header, join(ensemble_prob_path, f))
                np.savez(join(ensemble_prob_path,npz_name), probabilities=prob_map)
            else:
                exist_path_file = join(ensemble_prob_path, npz_name)
                exist_pred_npz = np.load(exist_path_file) #Load inference 
                exist_prob_map = exist_pred_npz['probabilities'] # Load probability maps
                # maybe increase precision to prevent rounding errors
                if exist_prob_map.dtype != np.float32:
                    exist_prob_map = exist_prob_map.astype(np.float32)
                new_prob_map = prob_map + exist_prob_map
                #new_target_image_with_header = nib.Nifti1Image(new_prob_map, pred_nii.affine, header=pred_nii.header)
                #nib.save(new_target_image_with_header, join(ensemble_prob_path, f))
                np.savez(join(ensemble_prob_path,npz_name), probabilities=new_prob_map)


def convert_prob_to_label(prob_folder, ensemble_label_path, input_folder_path):
    """
    Converts prob maps to labels, considering regions of nnUNet.
    #IN:
        prob_folder: Folder with the probability maps
        ensemble_label_path: Folder to save the labels 
        input_folder_path: Folder containing the input for the nnUNet. For getting the metadata
    """
    prob_files_list = listdir(prob_folder)
    for prob_file in prob_files_list:
        exist_path_file = join(prob_folder, prob_file)
        predicted_probabilities = np.load(exist_path_file)['probabilities']
        
        if predicted_probabilities.shape[0] == 4:
            segmentation = np.zeros([predicted_probabilities[0].shape[0], predicted_probabilities[0].shape[1], predicted_probabilities[0].shape[2]], dtype=np.uint16)
            for i, c in enumerate([1,2,3,4]):
                segmentation[predicted_probabilities[i] > 0.5] = c
                
        elif predicted_probabilities.shape[0] == 5:
            segmentation = np.zeros([predicted_probabilities[0].shape[0], predicted_probabilities[0].shape[1], predicted_probabilities[0].shape[2]], dtype=np.uint16)
            for i, c in enumerate([1,2,3,4]):
                segmentation[predicted_probabilities[c] > 0.5] = c
        else:
            print(f"predicted_probabilities has a weird number of channels: {predicted_probabilities.shape}") 

        segmentation = segmentation.transpose(2,1,0)
        meta_data_file = f"{prob_file.split('.npz')[0]}_0000.nii.gz"
        file_pred_nii = join(input_folder_path, meta_data_file)
        pred_nii = nib.load(file_pred_nii)

        new_target_image_with_header = nib.Nifti1Image(segmentation, pred_nii.affine, header=pred_nii.header)
        nii_name = f"{prob_file.split('.npz')[0]}.nii.gz"
        nib.save(new_target_image_with_header, join(ensemble_label_path, nii_name))

def perform_inference_step(inference_folder, input_folder_nnunet, ensemble_code):
    """
    Performing inference in teh covnerted dataset.
    This step loads trained models one by one and runs the command nnUNetv2_predict for each.
    # IN:
        inference_folder: root Path to save the inferences.
        input_folder_nnunet: Path where the files for inference are saved
    # OUT:
        NONE
    """
    print(f"Number of files for inference: {len(listdir(input_folder_nnunet))}")

    # Running inference only for each individual experiment
    # List of paths to save the inferences for each model
    #############################################################################################################
    ################################# Here it does the nnUNet!!!!! ##############################################
    results_inference_path= []
    if "RNg" in ensemble_code:
        results_inference_path.append(join(inference_folder, 'Dataset241_BraTS_2024_Real', 'nnUNetTrainer__nnUNetPlans__3d_fullres'))
    if "rGNg" in ensemble_code:
        results_inference_path.append(join(inference_folder, 'Dataset242_BraTS_2024_rGANs', 'nnUNetTrainer__nnUNetPlans__3d_fullres'))
    ###### Swin Unetr #######
    if "RSg" in ensemble_code:
        results_inference_path.append(join(inference_folder, 'Dataset241_BraTS_2024_Real', 'nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres_SwinUNETR'))
    if "rGSg" in ensemble_code:
        results_inference_path.append(join(inference_folder, 'Dataset242_BraTS_2024_rGANs', 'nnUNetTrainer_SwinUNETR__nnUNetPlans__3d_fullres_SwinUNETR'))
  
    # Creation of a command for each model inference
    dict_all = {}
    for entry in results_inference_path:
        # Creating a folder to save the inferences for each model
        maybe_mkdir_p(entry)
        print(f"{entry} created")
        key = entry.split('/')[-2].split('_')[-1]+"__"+entry.split('/')[-1]
        name = entry.split('/')[-2]
        dict_all[key] = {'name': name, '-o': entry, '-config': entry.split('/')[-1].split('__')[-1], '-tr': entry.split('/')[-1].split('__')[-3], '-p': 'nnUNetPlans'}

    print("Use the commands:")
    all_commands = []
    
    for key in dict_all:
        for fold in range(5):
            command = f"nnUNetv2_predict -d {dict_all[key]['name']} -i {input_folder_nnunet} -o {dict_all[key]['-o']}/fold_{fold} -f {fold} -tr {dict_all[key]['-tr']} -c {dict_all[key]['-config']} -p {dict_all[key]['-p']} -device cuda --save_probabilities" 
            print(f"nnUNetv2_predict -d {dict_all[key]['name']} -i {input_folder_nnunet} -o {dict_all[key]['-o']}/fold_{fold}  -f {fold} -tr {dict_all[key]['-tr']} -c {dict_all[key]['-config']} -p {dict_all[key]['-p']} -device cuda --save_probabilities")
            print("####")
            all_commands.append(command)
            
    #############################################################################################################
    ################################# Here it does the MedNeXt!!!!! ##############################################
    results_inference_path_mednext = []
    if "RMg" in ensemble_code:
        results_inference_path_mednext.append(join(inference_folder, 'Task241_BraTS_2024_Real', 'nnUNetTrainerV2_MedNeXt_L_kernel5__nnUNetPlansv2.1_trgSp_1x1x1__3d_fullres'))
    if "rGMg" in ensemble_code:
        results_inference_path_mednext.append(join(inference_folder, 'Task242_BraTS_2024_rGANs', 'nnUNetTrainerV2_MedNeXt_L_kernel5__nnUNetPlansv2.1_trgSp_1x1x1__3d_fullres'))
    
    dict_all_mednext = {}
    for entry in results_inference_path_mednext:
        # Creating a folder to save the inferences for each model
        maybe_mkdir_p(entry)
        print(f"{entry} created")
        key = entry.split('/')[-2].split('_')[-1]+"__"+entry.split('/')[-1]
        name = entry.split('/')[-2]
        print(f"key:{key}")
        print(f"name:{name}")
        dict_all_mednext[key] = {'name': name, '-o': entry, '-config': entry.split('/')[-1].split('__')[-1], '-tr': entry.split('/')[-1].split('__')[-3], '-p': 'nnUNetPlansv2.1_trgSp_1x1x1'}
        

        
    for key in dict_all_mednext:
        for fold in range(5):
            command = f"mednextv1_predict -t {dict_all_mednext[key]['name']} -i {input_folder_nnunet} -o {dict_all_mednext[key]['-o']}/fold_{fold} -f {fold} -tr {dict_all_mednext[key]['-tr']} -m {dict_all_mednext[key]['-config']} -p {dict_all_mednext[key]['-p']} -chk model_best --save_npz" 
            print(f"mednextv1_predict -t {dict_all_mednext[key]['name']} -i {input_folder_nnunet} -o {dict_all_mednext[key]['-o']}/fold_{fold}  -f {fold} -tr {dict_all_mednext[key]['-tr']} -m {dict_all_mednext[key]['-config']} -p {dict_all_mednext[key]['-p']} -chk model_best --save_npz")
            print("####")
            all_commands.append(command)
            
    
        
    
    # Very slow step
    # Running all inferences, one by one. 
    # One by one should be faster than doing it in parallel (No RAM problems neither slow writting)
    ensemble_prob_path = join(inference_folder, 'ensemble', ensemble_code, 'prob')
    maybe_mkdir_p(ensemble_prob_path)

    number_of_models = len(ensemble_code.split("_")) * 5
    print(f"Number of models for ensemble: {number_of_models}")
    first_infer = True
    
    for command_to_run in all_commands:
        print(f" Running command: {command_to_run}")
        print("__")
        result = subprocess.run(command_to_run, shell=True, text=True, capture_output=True) 
        print("Command output:")
        print(result.stdout) 
        inference_output_folder = command_to_run.split("-o ")[-1].split(" -f")[0]
        ensemble_predictions(inference_folder=inference_output_folder, ensemble_prob_path=ensemble_prob_path, number_models=number_of_models, first_infer=first_infer)
        #shutil.rmtree(inference_output_folder) 
        first_infer = False

    # Creating labels from prob
    ensemble_label_path = join(inference_folder, 'ensemble', ensemble_code, 'raw')
    maybe_mkdir_p(ensemble_label_path)
    convert_prob_to_label(prob_folder=ensemble_prob_path, ensemble_label_path=ensemble_label_path, input_folder_path=input_folder_nnunet)
    


################################################################################
# DOING THRESHOLD
################################################################################
def filter_structures_by_volume(pred_mat_cc, min_volume_threshold):
    """
    Removing the structures smaller than "min_volume_threshold".
    #IN:
        pred_mat_cc: Matrix where each individual structure have assigned a unique integer.
        min_volume_threshold: Threshold volume.
    """
    pred_label_cc = pred_mat_cc

    # Step 1: Count the frequency of each unique value in the array
    unique_values, value_counts = np.unique(pred_label_cc, return_counts=True)

    # Step 2: Compute the volume for each value
    voxel_volume = 1  # Assume voxel volume is 1 (you can adjust this value if needed)
    volumes_per_value = value_counts * voxel_volume 
    #print(f"volumes_per_value: {volumes_per_value}")

    # Step 3: Create another array without the values with volume less than 50
    selected_values = unique_values[volumes_per_value >= min_volume_threshold]

    # Step 4: Filter out the values with volume less than 50 from the original array
    filtered_array = np.where(np.isin(pred_label_cc, selected_values), pred_label_cc, 0)
    return filtered_array

def get_pred_region(prediction_matrix, tissue_type):
    """
    Get regions from the labels.
    This functions is adapted for the nnUNet convention.
    #IN:
       prediction_matrix: Numpy array (matrix) of the prediction.
       tissue_type: What region to return
    #OUT:
        prediction_matrix: Numpy array of the chosen region
    """
    
    if tissue_type == 'WT':
        np.place(prediction_matrix, (prediction_matrix != 1) & (prediction_matrix != 2) & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
    
    elif tissue_type == 'TC':
        np.place(prediction_matrix, (prediction_matrix != 2)  & (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)

        
    elif tissue_type == 'ET':
        np.place(prediction_matrix, (prediction_matrix != 3), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
    
    elif tissue_type == 'RC':
        np.place(prediction_matrix, (prediction_matrix != 4), 0)
        np.place(prediction_matrix, (prediction_matrix > 0), 1)
        
    return prediction_matrix


def from_region_to_label_non_conditional(filtered_array_regions):
    """
    Convert from region to label, considering the nnUNet convention
    #IN:
        filtered_array_regions: List of the three regions [WT,TC,ET], where each element is a numpy array.
    #OUT:
        new_array: Array with all labels.
    """
    new_array = np.zeros_like(filtered_array_regions[0])
    new_array[filtered_array_regions[0] != 0] = 1
    new_array[filtered_array_regions[1] != 0] = 2
    new_array[filtered_array_regions[2] != 0] = 3
    new_array[filtered_array_regions[3] != 0] = 4
    return new_array

def get_connected_components(prediction_seg, label_value):
    """
    Assigning unique integer values for each structure.
    #IN:  
        prediction_seg: Path to the segmentation predicted.
        label_value: Region.
    #OUT:
        pred_mat_cc:  Matrix where each individual structure have assigned a unique integer.
        pred_nii: Nifti file.
    """

    ## Get Prediction and GT segs matrix files
    pred_nii = nib.load(prediction_seg)
    pred_mat = pred_nii.get_fdata()
    
    pred_mat = get_pred_region(prediction_matrix = pred_mat,
                                tissue_type = label_value
                            )
    
    pred_mat_cc = cc3d.connected_components(pred_mat, connectivity=26)
    return pred_mat_cc, pred_nii

def process_prediction_file(prediction_seg, out_dir, min_volume_threshold_list):
    """
    Performing threshold in a specific file (prediction).
    #IN: 
        prediction_seg: Path to the file to be processed.
        out_dir: Path to save the new file
        min_volume_threshold_list: Threshold volume.
    #OUT:
        NONE
    """
    label_values = ['WT', 'TC', 'ET', 'RC']
    filtered_array_regions = []
    for index, region in enumerate(label_values):
        pred_mat_cc, pred_nii = get_connected_components(prediction_seg=prediction_seg, label_value=region)
        filtered_array = filter_structures_by_volume(pred_mat_cc=pred_mat_cc, min_volume_threshold=min_volume_threshold_list[index])
        filtered_array_regions.append(filtered_array)
    final_label = from_region_to_label_non_conditional(filtered_array_regions)
    target_image_with_header = nib.Nifti1Image(final_label, pred_nii.affine, header=pred_nii.header)
    nib.save(target_image_with_header, join(out_dir, prediction_seg.split('/')[-1]))


def thresholding_step(min_volume_threshold_WT, min_volume_threshold_TC, min_volume_threshold_ET, min_volume_threshold_RC, inference_folder, ensemble_code):
    """
    Performing thresholding step.
    #IN:
        min_volume_threshold_WT: Threshold volume for the WT region.
        min_volume_threshold_TC: Threshold volume for the TC region.
        min_volume_threshold_ET: Threshold volume for the ET region.
        min_volume_threshold_RC: Threshold volume for the RC region.
        inference_folder: Path for the inference folder.
        ensemble_code: Ensemble code
    #OUT:
        NONE
    """
    # Path to sabe the inference with threshold
    out_dir = join(inference_folder, "ensemble", ensemble_code, f"WT{min_volume_threshold_WT}_TC{min_volume_threshold_TC}_ET{min_volume_threshold_ET}_RC{min_volume_threshold_RC}")
    maybe_mkdir_p(out_dir)
    print(f"out_dir: {out_dir}")

    #Path for the ensemble inference, with no threshold
    ensemble_path = join(inference_folder, "ensemble", ensemble_code,  "raw") 
    ensemble_path_list = listdir(ensemble_path)
    ensemble_path_list.sort()

    print(f"Doing threshold: {ensemble_path_list}")
    print(f"Number of files in (ensemble_path_list): {len(ensemble_path_list)}")

    # Running processing in parallel
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = []
        for file_pref in ensemble_path_list:
            prediction_seg = join(ensemble_path, file_pref)
            min_volume_threshold_list = [min_volume_threshold_WT, min_volume_threshold_TC,
                                            min_volume_threshold_ET, min_volume_threshold_RC]
            future = executor.submit(process_prediction_file, prediction_seg, out_dir,
                                        min_volume_threshold_list)
            futures.append(future) 

        # Wait for all tasks to complete
        concurrent.futures.wait(futures)
    print(f"files after thresholding {listdir(out_dir)}")


################################################################################
# CONVERT BACK TO BRATS2024
################################################################################
def convert_labels_back_to_BraTS(seg: np.ndarray):
    """
    BraTS2023 changed the labels to RC - label 4, ET — label 3, ED — label 2 and NCR — label 1
    The input of the nnUNet requires labels to be sequencial, so the ED — label 2 was replaced with label 1 and NCR — label 1 with label 2 for training 
    Now we have to change it back.s
    #IN:
        seg: Numpy array of the segmentation
    #OUT:
        new_seg: Numpy array with values corrected
    """
    new_seg = np.zeros_like(seg)
    new_seg[seg == 1] = 2
    new_seg[seg == 3] = 3
    new_seg[seg == 2] = 1
    new_seg[seg == 4] = 4
    return new_seg

def load_convert_save(filename, input_folder, output_folder):
    """
    Loads the segmentation, performs convertion and save it back.
    #IN:
       filename: Name of the segmentation.
       input_folder: Path to the folder that contains the inferences from nnUNet.
       output_folder: Path where to save the converted segmentations.
    #OUT:
        NONE
    """
    a = sitk.ReadImage(join(input_folder, filename))
    b = sitk.GetArrayFromImage(a)
    c = convert_labels_back_to_BraTS(b)
    d = sitk.GetImageFromArray(c)
    d.CopyInformation(a)
    sitk.WriteImage(d, join(output_folder, filename))
    
def convert_labels_back_to_BraTS_2023_convention_folder(input_folder: str, output_folder: str, num_processes: int = 8):
    """
    reads all prediction files (nifti) in the input folder, converts the labels back to BraTS convention and saves the
    result in output_folder
    #IN:
        input_folder: Folder path with the final preiction from the pipeline with the nnUNet convention.
        output_folder: Folder path to save the converted semgentation to the BraTS2023
    #OUT:
        NONE
    """
    maybe_mkdir_p(output_folder)
    nii = subfiles(input_folder, suffix='.nii.gz', join=False)
    p = Pool(num_processes)
    p.starmap(load_convert_save, zip(nii, [input_folder] * len(nii), [output_folder] * len(nii)))
    p.close()
    p.join()

def convert_back_BraTS_step(min_volume_threshold_WT, min_volume_threshold_TC, min_volume_threshold_ET, min_volume_threshold_RC, inference_folder, ensemble_code, brats_final_inference):
    """
    Final step, where the files from all inference pipeline are converted back to BraTS2023 convention, as saved in the folder provided by the organizers.
    #IN:
        min_volume_threshold_WT: Threshold volume for the WT region.
        min_volume_threshold_TC: Threshold volume for the TC region.
        min_volume_threshold_ET: Threshold volume for the ET region.
        min_volume_threshold_RC: Threshold volume for the RC region.
        inference_folder: Path for the inference folder.
        ensemble_code: Ensemble code.
        brats_final_inference: Path to save the final segmentation (in BraTS2023 convention already).
    """
    #maybe_mkdir_p(brats_final_inference)

    input_folder = f'{inference_folder}/ensemble/{ensemble_code}/WT{min_volume_threshold_WT}_TC{min_volume_threshold_TC}_ET{min_volume_threshold_ET}_RC{min_volume_threshold_RC}'
    output_folder = f'{brats_final_inference}/'
    print(f"output_folder: {output_folder}")
    convert_labels_back_to_BraTS_2023_convention_folder(input_folder=input_folder, output_folder=output_folder, num_processes=8)
    print(f"Files in output folder: {listdir(output_folder)}")
    # NO need tp make a zip for the challenge.
    #folder_out_zip = output_folder 
    #shutil.make_archive(f"{folder_out_zip}", 'zip', output_folder)
    print("ALL DONE")

