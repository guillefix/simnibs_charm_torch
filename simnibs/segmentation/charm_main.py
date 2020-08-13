# -*- coding: utf-8 -*-
"""
Created on Sun Feb  2 17:12:24 2020

@author: axthi
"""


import logging
import os
import shutil
import time
from .. import utils
from simnibs import SIMNIBSDIR
from ..utils.simnibs_logger import logger
from ..utils import file_finder
from ..utils import transformations
#from . import samseg
import nibabel as nib
from ._cat_c_utils import sanlm
import numpy as np
from scipy import ndimage
from ..utils.transformations import resample_vol
from ..mesh_tools import meshing
from . import _thickness
from ..mesh_tools.mesh_io import write_msh
from .brain_surface import createCS
import multiprocessing
import functools


def _register_atlas_to_input_affine(T1, template_file_name,
                                    affine_mesh_collection_name, mesh_level1,
                                    mesh_level2, save_path,
                                    template_coregistered_name,
                                    init_atlas_settings, neck_tissues,
                                    visualizer,
                                    init_transform=None,
                                    world_to_world_transform_matrix=None):

    # Import the affine registration function
    scales = init_atlas_settings['affine_scales']
    thetas = init_atlas_settings['affine_rotations']
    thetas_rad = [theta * np.pi/180 for theta in thetas]
    neck_search_bounds = init_atlas_settings['neck_search_bounds']
    ds_factor = init_atlas_settings['dowsampling_factor_affine']
    affine = samseg.AffineWholeHead(T1, affine_mesh_collection_name,
                                    template_file_name)

    init_options = samseg.initializationOptions(pitchAngles=thetas_rad,
                                               scales=scales)
    
    image_to_image_transform, world_to_world_transform, optimization_summary =\
    affine.registerAtlas(initializationOptions=init_options,
                         targetDownsampledVoxelSpacing=ds_factor,
                         visualizer=visualizer)
 
    affine.saveResults(T1, template_file_name, save_path,
                       template_coregistered_name, image_to_image_transform,
                       world_to_world_transform)

    logger.info('Template registration summary.')
    logger.info('Number of Iterations: %d, Cost: %f\n' %
                (optimization_summary['numberOfIterations'],
                 optimization_summary['cost']))

    logger.info('Adjusting neck.')
    affine.adjust_neck(T1, template_coregistered_name, mesh_level1,
                       mesh_level2, neck_search_bounds, neck_tissues,
                       visualizer)
    logger.info('Neck adjustment done.')


def _denoise_input_and_save(input_name, output_name):
    input_raw = nib.load(input_name)
    img = input_raw.get_data()
    # Sometimes the images have an extra unit dimension,
    # squeeze that out if it's there.
    img = img.squeeze()
    img_smoothed = sanlm(img, 3, 1)
    output_smoothed = nib.Nifti1Image(img_smoothed, input_raw.affine)
    nib.save(output_smoothed, output_name)


def _estimate_parameters(path_to_segment_folder,
                         template_coregistered_name,
                         path_to_atlas_folder, input_images,
                         visualizer):

    # TODO: Should the model specs be here or in .ini?
    user_optimization_options = {'multiResolutionSpecification':
                                 [{'atlasFileName':
                                   os.path.join(path_to_segment_folder,
                                                'atlas_level1.txt.gz'),
                                   'targetDownsampledVoxelSpacing': 2.0,
                                   'maximumNumberOfIterations': 100,
                                   'estimateBiasField': True},
                                  {'atlasFileName':
                                      os.path.join(path_to_segment_folder,
                                                   'atlas_level2.txt.gz'),
                                   'targetDownsampledVoxelSpacing': 1.0,
                                   'maximumNumberOfIterations': 100,
                                   'estimateBiasField': True}]}

    user_model_specifications = {'atlasFileName': os.path.join(
            path_to_segment_folder, 'atlas_level2.txt.gz'),
                                 'biasFieldSmoothingKernelSize': 70,
                                 'brainMaskingThreshold': 0.01}
    if len(input_images) > 1:
        user_model_specifications = {'atlasFileName': os.path.join(
            path_to_segment_folder, 'atlas_level2.txt.gz'),
                                 'biasFieldSmoothingKernelSize': 50,
                                 'brainMaskingThreshold': 0.01}


    #TODO: THIS IS TO MAKE TESTING FASTER, REMOVE LATER
    if 0:
        user_optimization_options = {'multiResolutionSpecification':
                                     [{'atlasFileName':
                                       os.path.join(path_to_segment_folder,
                                                    'atlas_level1.txt.gz'),
                                       'targetDownsampledVoxelSpacing': 3.0,
                                       'maximumNumberOfIterations': 10,
                                       'estimateBiasField': True},
                                      {'atlasFileName':
                                          os.path.join(path_to_segment_folder,
                                                       'atlas_level2.txt.gz'),
                                       'targetDownsampledVoxelSpacing': 2.0,
                                       'maximumNumberOfIterations': 10,
                                       'estimateBiasField': True}]}

    samseg_kwargs = dict(
        imageFileNames=input_images,
        atlasDir=path_to_atlas_folder,
        savePath=path_to_segment_folder,
        transformedTemplateFileName=template_coregistered_name,
        userModelSpecifications=user_model_specifications,
        userOptimizationOptions=user_optimization_options,
        visualizer=visualizer,
        saveHistory=False,
        saveMesh=False,
        savePosteriors=False,
        saveWarp=True)

    logger.info('Starting segmentation.')
    samsegment = samseg.SamsegWholeHead(**samseg_kwargs)
    samsegment.preProcess()
    samsegment.process()

    # Print optimization summary
    optimizationSummary = samsegment.getOptimizationSummary()
    for multiResolutionLevel, item in enumerate(optimizationSummary):
        logger.info('atlasRegistrationLevel%d %d %f\n' %
                    (multiResolutionLevel, item['numberOfIterations'],
                     item['perVoxelCost']))
        
    return samsegment.saveParametersAndInput()


def _post_process_segmentation(bias_corrected_image_names,
                               upsampled_image_names,
                               tissue_settings,
                               parameters_and_inputs,
                               transformed_template_name):

    logger.info('Upsampling bias corrected images.')
    for input_number, bias_corrected in enumerate(bias_corrected_image_names):
        corrected_input = nib.load(bias_corrected)
        resampled_input, new_affine, orig_res = resample_vol(
                                                    corrected_input.get_data(),
                                                    corrected_input.affine,
                                                    0.5, order=3)
        upsampled = nib.Nifti1Image(resampled_input, new_affine)
        nib.save(upsampled, upsampled_image_names[input_number])

    # Next we need to reconstruct the segmentation with the upsampled data
    # and map it into the simnibs tissues
    upsampled_tissues = samseg.simnibs_segmentation_utils.segmentUpsampled(
                        upsampled_image_names,
                        tissue_settings,
                        parameters_and_inputs,
                        transformed_template_name)

    # Do morphological operations
    simnibs_tissues = tissue_settings['simnibs_tissues']
    _morphological_operations(upsampled_tissues, simnibs_tissues)

    return upsampled_tissues


def _morphological_operations(label_img, simnibs_tissues):
    ''' Does morphological operations to ensure
        1. A CSF layer between GM and Skull and beteween GM and CSF
        2. Outer bone layers are compact bone

        works in-place
    '''
    # Ensuring a CSF layer beteen GM and Skull and GM and Blood
    # Relabel regions in the expanded GM which are in skull or blood to CSF
    GM = label_img == simnibs_tissues["GM"]
    C_BONE = label_img == simnibs_tissues["Compact_bone"]
    S_BONE = label_img == simnibs_tissues["Spongy_bone"]
    BLOOD = label_img == simnibs_tissues["Blood"]
    GM_dilated = ndimage.morphology.binary_dilation(
        GM, ndimage.generate_binary_structure(3, 2)
    )
    label_img[GM_dilated * (C_BONE + S_BONE + BLOOD)] = simnibs_tissues["CSF"]
    # Ensure the outer skull label is compact bone
    C_BONE = label_img == simnibs_tissues["Compact_bone"]
    S_BONE = label_img == simnibs_tissues["Spongy_bone"]
    SKULL_outer = (C_BONE + S_BONE) * ~ndimage.morphology.binary_erosion(
        (C_BONE + S_BONE), ndimage.generate_binary_structure(3, 2)
    )
    label_img[SKULL_outer] = simnibs_tissues["Compact_bone"]


def view(subject_dir):
    print('charm viewer not yet implemented, sorry...')


def run(subject_dir=None, T1=None, T2=None,
        registerT2=False, initatlas=False, segment=False,
        create_surfaces=True, mesh_image=False, skipregisterT2=False,
        usesettings=None, options_str=None):
    """charm pipeline

    PARAMETERS
    ----------
    subject_dir : str, mandatory
        output directory
    T1 : str
        filename of T1 image
    T2 : str
        filename of T2 image

    --> parameters to control the workflow:
    registerT2 : bool
        run T2-to-T1 registration (default = False)
    initatlas : bool
        run affine registration of atlas to input images (default = False)
    segment : bool
        run volume and surface segmentation (default = False)
    mesh_image : bool
        run tetrahedral meshing (default = False)
    --> further parameters:
    skipregisterT2 : bool
    skip T2-to-T1 registration, just copy T2-weighted image (default = False)
    usesettings : str
        filename of alternative settings-file (default = None)
    options_str : str
        string of command line options to add to logging (default = None)
    RETURNS
    ----------
        None
    """
    # ------------------------START UP-----------------------------------------
    start = time.time()

    # make subject_dir if not existent
    if not os.path.exists(subject_dir):
        os.mkdir(subject_dir)

    # start logging ...
    logfile = os.path.join(subject_dir, "charm_log.html")
    with open(logfile, 'a') as f:
        f.write('<HTML><HEAD><TITLE>charm report</TITLE></HEAD><BODY><pre>')
        f.close()
    fh = logging.FileHandler(logfile, mode='a')
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    fh.setFormatter(formatter)
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    utils.simnibs_logger.register_excepthook(logger)

    if options_str is not None:
        logger.debug('options: '+options_str)
    logger.info('charm run started: '+time.asctime())

    # initialize subject files
    sub_files = file_finder.SubjectFiles(None, subject_dir)

    # get subID
    subID = sub_files.subid

    # copy T1 (as nii.gz) if supplied
    if T1 is not None:
        if not os.path.exists(T1):
            raise FileNotFoundError(f'Could not find input T1 file: {T1}')
        if len(T1) > 7 and T1[-7:].lower() == '.nii.gz':
            shutil.copyfile(T1, sub_files.T1)
        else:
            nib.save(nib.load(T1), sub_files.T1)

    if skipregisterT2 and T2 is not None:
        # skip T2-to-T1 registration, just copy T2 image (as nii.gz)
        registerT2 = False
        if not os.path.exists(T2):
            raise FileNotFoundError(f'Could not find input T2 file: {T2}')
        if len(T2) > 7 and T2[-7:].lower() == '.nii.gz':
            shutil.copyfile(T2, sub_files.T2_reg)
        else:
            nib.save(nib.load(T2), sub_files.T2_reg)

    # read settings and copy settings file
    if usesettings is None:
        fn_settings = os.path.join(SIMNIBSDIR, 'charm.ini')
    else:
        fn_settings = usesettings
    settings = utils.settings_reader.read_ini(fn_settings)
    shutil.copyfile(fn_settings, sub_files.settings)
    # -------------------------PIPELINE STEPS---------------------------------
    # TODO: denoise T1 here with the sanlm filter, T2 denoised after coreg.
    # Could be before as well but doesn't probably matter that much.
    # Denoising after has the benefit that we keep around the non-denoised
    # versions of all inputs.

    # Make the output path for segmentations
    os.makedirs(sub_files.segmentation_folder, exist_ok=True)

    denoise_settings = settings['preprocess']
    if denoise_settings['denoise']:
        logger.info('Denoising the T1 input and saving.')
        _denoise_input_and_save(T1, sub_files.T1_denoised)

    if registerT2:
        # register T2 to T1
        logger.info('starting registerT2')

        # get local settings
        local_settings=settings['registerT2']

        # check input files
        if not os.path.exists(sub_files.T1):
            raise FileNotFoundError(f'Could not find subject T1 file')
        if not os.path.exists(sub_files.T2_reg):
            raise FileNotFoundError(f'Could not find subject T2 file')

        # do your stuff

        # write QA results

    # denoise if needed
    if denoise_settings['denoise'] and os.path.exists(sub_files.T2_reg):
        logger.info('Denoising the registered T2 and saving.')
        _denoise_input_and_save(sub_files.T2_reg,
                                sub_files.T2_reg_denoised)

    # Set-up samseg related things before calling the affine registration
    # and/or segmentation
    samseg_settings = settings['samseg']

    # Specify the maximum number of threads the GEMS code will use
    # by default this is all, but can be changed in the .ini
    num_threads = samseg_settings['threads']
    if isinstance(num_threads, int) and num_threads > 0:
        samseg.setGlobalDefaulNumberOfThreads(num_threads)
        logger.info('Using %d threads, instead of all available.'
                    % num_threads)

    # TODO: Setup the visualization tool. This needs some pyqt stuff to be
    # installed. Don't know if we want to expose this in the .ini
    showFigs = False
    showMovies = False
    visualizer = samseg.initVisualizer(showFigs, showMovies)

    # Set-up atlas paths
    atlas_name = samseg_settings['atlas_name']
    logger.info('Using '+atlas_name+' as charm atlas.')
    atlas_path = os.path.join(file_finder.templates.charm_atlas_path,
                              atlas_name)
    atlas_settings = utils.settings_reader.read_ini(
            os.path.join(atlas_path, atlas_name+'.ini'))
    atlas_settings_names = atlas_settings['names']
    template_name = os.path.join(atlas_path,
                                 atlas_settings_names['template_name'])
    atlas_affine_name = os.path.join(atlas_path,
                                     atlas_settings_names['affine_atlas'])
    atlas_level1 = os.path.join(atlas_path,
                                atlas_settings_names['atlas_level1'])
    atlas_level2 = os.path.join(atlas_path,
                                atlas_settings_names['atlas_level2'])
    neck_tissues = atlas_settings['neck_optimization']
    neck_tissues = neck_tissues['neck_tissues']

    if initatlas:
        # initial affine registration of atlas to input images,
        # including break neck
        logger.info('Starting affine registration and neck correction.')
        if denoise_settings['denoise']:
            input_image = sub_files.T1_denoised
        else:
            input_image = T1

        init_atlas_settings = settings['initatlas']
        _register_atlas_to_input_affine(input_image, template_name,
                                        atlas_affine_name,
                                        atlas_level1,
                                        atlas_level2,
                                        sub_files.segmentation_folder,
                                        sub_files.template_coregistered,
                                        init_atlas_settings,
                                        neck_tissues,
                                        visualizer)

    if segment:
        # This part runs the segmentation, upsamples bias corrected output,
        # writes mni transforms, creates upsampled segmentation, maps tissues
        # to conductivities, runs morphological operations

        # Run the segmentation and return the class, which is needed
        # for further post-processing
        # The bias field kernel size has to be changed based on input
        input_images = []
        if denoise_settings['denoise']:
            input_images.append(sub_files.T1_denoised)
        else:
            input_images.append(T1)

        if os.path.exists(sub_files.T2_reg):
            if denoise_settings['denoise']:
                input_images.append(sub_files.T2_reg_denoised)
            else:
                input_images.append(sub_files.T2_reg)

        logger.info('Estimating parameters.')
        segment_parameters_and_inputs = _estimate_parameters(
                sub_files.segmentation_folder,
                sub_files.template_coregistered,
                atlas_path, input_images,
                visualizer)

        # Okay now the parameters have been estimated, and we can segment the
        # scan. However, we need to also do this at an upsampled resolution,
        # so first write out the bias corrected scan, and the segmentation.

        bias_corrected_image_names = [sub_files.T1_bias_corrected]
        if len(input_images) > 1:
            bias_corrected_image_names.append(sub_files.T2_bias_corrected)

        logger.info('Writing out normalized images and labelings.')
        if create_surfaces:
            os.makedirs(sub_files.surface_folder, exist_ok=True)
            cat_images = [sub_files.norm_image,
                          sub_files.cereb_mask,
                          sub_files.subcortical_mask,
                          sub_files.parahippo_mask,
                          sub_files.hemi_mask]

            cat_structs = atlas_settings['CAT_structures']
            samseg.simnibs_segmentation_utils.writeBiasCorrectedImagesAndSegmentation(
                            bias_corrected_image_names,
                            sub_files.labeling,
                            segment_parameters_and_inputs,
                            cat_structure_options=cat_structs,
                            cat_images=cat_images)
        else:
            samseg.simnibs_segmentation_utils.writeBiasCorrectedImagesAndSegmentation(
                            bias_corrected_image_names,
                            sub_files.labeling,
                            segment_parameters_and_inputs)

        # Write out MNI warps
        if 0:
            logger.info('Writing out MNI warps.')
            os.makedirs(sub_files.mni_transf_folder, exist_ok=True)
            samseg.simnibs_segmentation_utils.saveWarpField(
                    template_name,
                    sub_files.conf2mni_nonl,
                    sub_files.mni2conf_nonl,
                    segment_parameters_and_inputs)

        # Run post-processing
        logger.info('Post-processing segmentation')
        os.makedirs(sub_files.label_prep_folder, exist_ok=True)

        upsampled_image_names = [sub_files.T1_upsampled]
        if len(bias_corrected_image_names) > 1:
            upsampled_image_names.append(sub_files.T2_upsampled)

        tissue_settings = atlas_settings['conductivity_mapping']
        cleaned_upsampled_tissues = _post_process_segmentation(
                                    bias_corrected_image_names,
                                    upsampled_image_names,
                                    tissue_settings,
                                    segment_parameters_and_inputs,
                                    sub_files.template_coregistered)

        # Write to disk
        upsampled_image = nib.load(sub_files.T1_upsampled)
        affine_upsampled = upsampled_image.affine
        upsampled_tissues = nib.Nifti1Image(cleaned_upsampled_tissues,
                                            affine_upsampled)
        nib.save(upsampled_tissues, sub_files.tissue_labeling_upsampled)

    if create_surfaces:
        # Create surfaces ala CAT12
        logger.info('Starting surface creation')
        fsavgDir = file_finder.Templates().templates_surfaces

        # input images and image header 
        Ymf = nib.load(sub_files.norm_image)
        vox2mm = Ymf.affine
        Yleft = nib.load(sub_files.hemi_mask)
        Ymaskhemis = nib.load(sub_files.cereb_mask)
        Ymaskparahipp = nib.load(sub_files.parahippo_mask)

        # parameters (will be put into the .ini file)
        surf=['lh','rh','lc','rc']
        #surf = ['lc', 'rc']
        vdist = [1.0, 0.75]
        voxsize_pbt = [0.5, 0.25]
        voxsize_refineCS = [0.75, 0.5]
        no_selfintersections = True
        th_initial = 0.714

        # parameters that will we likely skip...
        add_parahipp = 0
        close_parahipp = False

        starttime = time.time()
        Pcentral_all = []
        Pspherereg_all = []
        Pthick_all = []
        EC_all = []
        defect_size_all = []
        # GBS: changed here
        for si in range(len(surf)):
            Pcentral, Pspherereg, Pthick, EC, defect_size = createCS(
                                                            Ymf.get_data(),
                                                            Yleft.get_data(),
                                                            Ymaskhemis.get_data(),
                                                            Ymaskparahipp.get_data(),
                                                            vox2mm,
                                                            surf[si],
                                                            sub_files.surface_folder,
                                                            fsavgDir,
                                                            vdist,
                                                            voxsize_pbt,
                                                            voxsize_refineCS,
                                                            th_initial,
                                                            no_selfintersections,
                                                            add_parahipp,
                                                            close_parahipp)

            Pcentral_all.append(Pcentral)
            Pspherereg_all.append(Pspherereg)
            Pthick_all.append(Pthick)
            EC_all.append(EC)
            defect_size_all.append(defect_size)
            
        # print time duration
        elapsed = time.time() - starttime    
        print('\nTotal time cost (HH:MM:SS):')
        print(time.strftime('%H:%M:%S', time.gmtime(elapsed)))

    if mesh_image:
        # create mesh from label image
        logger.info('Starting mesh')
        label_image = nib.load(sub_files.tissue_labeling_upsampled)
        label_buffer = label_image.get_data()
        # Cast to uint16, otherwise meshing complains
        label_buffer = label_buffer.astype(np.uint16)
        label_affine = label_image.affine
        # Read in settings for meshing
        mesh_settings = settings['mesh']
        size_slope = mesh_settings['size_slope']
        size_range = mesh_settings['size_range']
        distance_slope = mesh_settings['distance_slope']
        distance_range = mesh_settings['distance_range']
        optimize = mesh_settings['optimize']
        remove_spikes = mesh_settings['remove_spikes']
        smooth_steps = mesh_settings['smooth_steps']
        final_mesh = mesh(label_buffer, label_affine, size_slope=size_slope,
                          size_range=size_range, distance_slope=distance_slope,
                          distance_range=distance_range, optimize=optimize,
                          remove_spikes=remove_spikes,
                          smooth_steps=smooth_steps)
        logger.info('Writing mesh')
        write_msh(final_mesh, sub_files.head_mesh)

    # -------------------------TIDY UP-----------------------------------------

    # log stopping time and total duration ...
    logger.info('charm run finished: '+time.asctime())
    logger.info('Total running time: '+utils.simnibs_logger.format_time(
                                                            time.time()-start))

    # stop logging ...
    while logger.hasHandlers():
        logger.removeHandler(logger.handlers[0])
    utils.simnibs_logger.unregister_excepthook()
    logging.shutdown()
    with open(logfile, 'a') as f:
        f.write('</pre></BODY></HTML>')
        f.close()


def mesh(label_img, affine, size_slope=1.0, size_range=(1, 5),
         distance_slope=0.5, distance_range=(0.1, 3),
         optimize=True, remove_spikes=True, smooth_steps=5):
    ''' Creates a mesh from a labeled image

    The maximum element sizes (CGAL facet_size and cell_size) is given by:
        size = size_slope * thickness
        size[size < size_range[0]] = size_range[0]
        size[size > size_range[1]] = size_range[1]

    where "thickness" is the local tissue thickness.
    The distance (CGAL facet_distance) parameter is calcualted in a similar way.
    This allows for the meshing to adjust sizes according to local needs.

    Parameters
    -----------
    label_img: 3D np.ndarray in uint8 format
        Labeled image from segmentation
    affine: 4x4 np.ndarray
        Affine transformation from voxel coordinates to world coordinates
    size_slope: float (optinal)
        relationship between thickness and slopes. The largest this value,
        the larger the elements will be. Default: 1
    size_range: 2-element list (optional)
        Minimum and maximum values for element sizes, in mm. Default: (1, 5)
    distance_slope: float (optinal)
        relationship between thickness and facet_distance. At small distance
        values, the meshing will follow the original image more strictly. This
        also means more elements. Default: 0.5
    size_range: 2-element list (optional)
        Minimum and maximum values for facet_distance, in mm. Default: (0.1, 3)
    optimize: bool (optional)
        Whether to run lloyd optimization on the mesh. Default: True
    remove_spikes: bool (optional)
        Whether to remove spikes to create smoother meshes. Default: True
    smooth_steps: int (optional)
        Number of smoothing steps to apply to the final mesh surfaces. Default: 5

    Returns
    --------
    msh: simnibs.Msh
        Mesh structure
    '''
    # Calculate thickness
    logger.info('Calculating tissue thickness')
    thickness = _thickness._calc_thickness(label_img)
    # I set the background thickness to some large value
    thickness[thickness < .5] = 100
    # Scale thickenss with voxel size
    voxel_size = transformations.get_vox_size(affine)
    if not np.allclose(np.diff(voxel_size), 0):
        logger.warn('Anisotropic image, meshing may contain extra artifacts')
    thickness *= np.average(voxel_size)

    # Define size field and distance field
    size_field = _sizing_field_from_thickness(
        thickness, size_slope, size_range
    )
    distance_field = _sizing_field_from_thickness(
        thickness, distance_slope, distance_range
    )
    # Run meshing
    logger.info('Meshing')
    start = time.time()
    mesh = meshing.image2mesh(
        label_img,
        affine,
        facet_size=size_field,
        facet_distance=distance_field,
        cell_size=size_field,
        optimize=optimize
    )
    logger.info(
        'Time to mesh: ' +
        utils.simnibs_logger.format_time(time.time()-start)
    )
    start = time.time()
    # Separate out tetrahedron (will reconstruct triangles later)
    mesh = mesh.crop_mesh(elm_type=4)
    # Assign the right labels to the mesh as CGAL modifies them
    indices_seg = np.unique(label_img)[1:]
    new_tags = np.copy(mesh.elm.tag1)
    for i, t in enumerate(indices_seg):
        new_tags[mesh.elm.tag1 == i+1] = t
    mesh.elm.tag1 = new_tags
    mesh.elm.tag2 = new_tags.copy()
    # Remove spikes from mesh
    if remove_spikes:
        logger.info('Removing Spikes')
        meshing.despike(
            mesh, relabel_tol=1e-5,
            adj_threshold=2
        )
    # Reconctruct the mesh surfaces
    logger.info('Reconstructing Surfaces')
    mesh.fix_th_node_ordering()
    mesh.reconstruct_surfaces()
    # Smooth the mesh
    if smooth_steps > 0:
        logger.info('Smoothing Mesh Surfaces')
        mesh.smooth_surfaces(smooth_steps, step_size=0.3, max_gamma=10)

    logger.info(
        'Time to post-process mesh: ' +
        utils.simnibs_logger.format_time(time.time()-start)
    )
    return mesh


def _sizing_field_from_thickness(thickness, slope, ranges):
    ''' Calculates a sizing field from thickness '''
    if ranges[0] > ranges[1]:
        raise ValueError('Ranges value should be in format (min, max)')
    field = np.array(slope*thickness, dtype=np.float32, order='F')
    field[field < ranges[0]] = ranges[0]
    field[field > ranges[1]] = ranges[1]
    return field
