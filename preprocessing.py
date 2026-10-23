import os
import pandas as pd
import numpy as np
import nibabel as nib
from tqdm import tqdm
import glob
from sklearn.preprocessing import StandardScaler

# Constants
AAT_TO_WAB_FORMULA = True  # Whether to convert AAT scores to WAB AQ scores

def convert_aat_to_wab(spontaneous, comprehension, repetition, naming):
    """
    Convert AAT scores to WAB AQ scores using the formula:
    AQ = (Spontaneous + Comprehension ÷ 20 + Repetition ÷ 10 + Naming ÷ 10) × 2
    
    Parameters:
    - spontaneous: Spontaneous speech score (20 points total)
    - comprehension: Comprehension score (200 points total)
    - repetition: Repetition score (100 points total)
    - naming: Naming score (100 points total)
    
    Returns:
    - AQ: The Aphasia Quotient (WAB score), normalized to 0-100 scale
    """
    aq = (spontaneous + comprehension / 20 + repetition / 10 + naming / 10) * 2
    return aq

def vec2mat(fc, fillones=True):
    """Convert vector to matrix"""
    d = int(round((1+(1+8*fc.size)**0.5)/2))
    a,b = np.triu_indices(d,1)
    mat = np.zeros((d,d))
    mat[a,b] = fc
    mat += mat.T
    ones = np.arange(d)
    if fillones:
        mat[ones,ones] = 1
    return mat

def mat2vec(fc):
    """Convert matrix to vector"""
    d = fc.shape[0]
    a,b = np.triu_indices(d,1)
    return fc[a,b]

def load_arc_data(base_dir="ARCData"):
    """Load resting state data from ARCData folder (no longitudinal data)"""
    # Load participants data
    participants_path = os.path.join(base_dir, "participants.tsv")
    participants_df = pd.read_csv(participants_path, sep='\t')
    
    # Get resting state fMRI files
    rest_files = glob.glob(os.path.join(base_dir, "*task-rest*.nii.gz"))
    
    # Extract subject IDs from filenames
    subject_ids = [os.path.basename(f).split('_')[0] for f in rest_files]
    
    # Process each resting state file and get functional connectivity
    fc_data = {}
    for subject_id, file_path in tqdm(zip(subject_ids, rest_files), desc="Processing ARC resting state data"):
        try:
            # Load the NIfTI file
            nii_img = nib.load(file_path)
            # Get time series data
            time_series = nii_img.get_fdata()
            # Reshape to (time, voxels)
            orig_shape = time_series.shape
            time_series = time_series.reshape(orig_shape[0]*orig_shape[1]*orig_shape[2], orig_shape[3])
            # Compute correlation matrix
            fc_matrix = np.corrcoef(time_series.T)
            # Store as vectorized upper triangle
            fc_data[subject_id] = mat2vec(fc_matrix)
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    # Merge FC data with participant info
    fc_df = pd.DataFrame.from_dict(fc_data, orient='index')
    
    # Process the participant data to make it consistent
    participants_df['subject_id'] = participants_df['participant_id']
    participants_df = participants_df.set_index('subject_id')
    
    # Include only subjects with FC data
    participants_df = participants_df.loc[fc_df.index]
    
    # Results dictionary
    results = {
        'fc_data': fc_df,
        'metadata': participants_df,
        'is_longitudinal': False,
        'is_resting_state': True,
        'source': 'ARCData'
    }
    
    return results

def load_figshare_data(base_dir="FigShare"):
    """Load task-based longitudinal data from FigShare folder"""
    # Load behavioral data with AAT scores
    behavioral_path = os.path.join(base_dir, "behavioral_data_LRS.csv")
    if os.path.exists(behavioral_path):
        try:
            behavioral_df = pd.read_csv(behavioral_path)
            # Apply AAT to WAB conversion formula
            if 'Spontaneous' in behavioral_df.columns and AAT_TO_WAB_FORMULA:
                behavioral_df['WAB_AQ'] = behavioral_df.apply(
                    lambda row: convert_aat_to_wab(
                        row.get('Spontaneous', 0),
                        row.get('Comprehension', 0),
                        row.get('Repetition', 0),
                        row.get('Naming', 0)
                    ),
                    axis=1
                )
        except Exception as e:
            print(f"Error loading behavioral data: {e}")
            behavioral_df = None
    else:
        print(f"Behavioral data file not found at {behavioral_path}")
        behavioral_df = None
    
    # Load demographic data
    demo_path = os.path.join(base_dir, "figshare_demo.csv")
    if os.path.exists(demo_path):
        demo_df = pd.read_csv(demo_path)
        # Standardize columns
        demo_df.rename(columns={'Patient ID': 'subject_id', 'Gender': 'sex'}, inplace=True)
        if 'subject_id' in demo_df.columns:
            demo_df['subject_id'] = demo_df['subject_id'].astype(str).str.zfill(2)
    else:
        print(f"Demo data file not found at {demo_path}")
        demo_df = None
    
    # Results dictionary
    results = {
        'behavioral_data': behavioral_df,
        'metadata': demo_df,
        'is_longitudinal': True,
        'is_resting_state': False,
        'source': 'FigShare'
    }
    
    return results

def load_osf_data(base_dir="OSFData"):
    """Load resting state longitudinal data from OSFData folder"""
    # Load behavioral data
    behavioral_path = os.path.join(base_dir, "behavioral_data.csv")
    if os.path.exists(behavioral_path):
        behavioral_df = pd.read_csv(behavioral_path)
    else:
        print(f"Behavioral data file not found at {behavioral_path}")
        behavioral_df = None
    
    # Load FC graph covariate data
    covariate_path = os.path.join(base_dir, "FC_graph_covariate_data.csv")
    if os.path.exists(covariate_path):
        covariate_df = pd.read_csv(covariate_path)
        covariate_df.rename(columns={'ID': 'subject_id'}, inplace=True)
    else:
        print(f"Covariate data file not found at {covariate_path}")
        covariate_df = None
    
    # Load resting state nifti files
    rest_files = glob.glob(os.path.join(base_dir, "P*_rs.nii"))
    
    # Process each resting state file and get functional connectivity
    fc_data = {}
    for file_path in tqdm(rest_files, desc="Processing OSF resting state data"):
        try:
            subject_id = os.path.basename(file_path).split('_')[0]
            # Load the NIfTI file
            nii_img = nib.load(file_path)
            # Get time series data
            time_series = nii_img.get_fdata()
            # Reshape to (time, voxels)
            orig_shape = time_series.shape
            time_series = time_series.reshape(orig_shape[0]*orig_shape[1]*orig_shape[2], orig_shape[3])
            # Compute correlation matrix
            fc_matrix = np.corrcoef(time_series.T)
            # Store as vectorized upper triangle
            fc_data[subject_id] = mat2vec(fc_matrix)
        except Exception as e:
            print(f"Error processing {subject_id}: {e}")
    
    # Convert FC data to DataFrame
    if fc_data:
        fc_df = pd.DataFrame.from_dict(fc_data, orient='index')
    else:
        fc_df = None
        print("No FC data was successfully processed")
    
    # Results dictionary
    results = {
        'fc_data': fc_df,
        'behavioral_data': behavioral_df,
        'metadata': covariate_df,
        'is_longitudinal': True,
        'is_resting_state': True,
        'source': 'OSFData'
    }
    
    return results

def standardize_features(df, columns=None):
    """Standardize features to mean=0, std=1"""
    if columns is None:
        columns = df.columns
    
    scaler = StandardScaler()
    df[columns] = scaler.fit_transform(df[columns])
    return df

def prepare_combined_dataset():
    """
    Prepare a combined dataset from all three sources for aphasia treatment prediction
    
    Returns:
        dict: Contains preprocessed data ready for model training
    """
    # Load data from all three sources
    arc_data = load_arc_data()
    figshare_data = load_figshare_data()
    osf_data = load_osf_data()
    
    # Combine metadata from all sources
    all_metadata = []
    
    # Process ARC data
    if arc_data['metadata'] is not None:
        arc_meta = arc_data['metadata'].reset_index()
        arc_meta['source'] = 'ARC'
        arc_meta['wab_aq'] = arc_meta['wab_aq']
        arc_meta['is_longitudinal'] = False
        all_metadata.append(arc_meta)
    
    # Process FigShare data
    if figshare_data['metadata'] is not None and figshare_data['behavioral_data'] is not None:
        figshare_meta = figshare_data['metadata']
        if 'WAB_AQ' in figshare_data['behavioral_data'].columns:
            # Merge behavioral data to get WAB_AQ scores
            figshare_meta = pd.merge(
                figshare_meta,
                figshare_data['behavioral_data'][['subject_id', 'WAB_AQ']],
                on='subject_id',
                how='left'
            )
            figshare_meta['wab_aq'] = figshare_meta['WAB_AQ']
        figshare_meta['source'] = 'FigShare'
        figshare_meta['is_longitudinal'] = True
        all_metadata.append(figshare_meta)
    
    # Process OSF data
    if osf_data['metadata'] is not None:
        osf_meta = osf_data['metadata']
        osf_meta['source'] = 'OSF'
        osf_meta['is_longitudinal'] = True
        all_metadata.append(osf_meta)
    
    # Combine all metadata
    if all_metadata:
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
    else:
        combined_metadata = pd.DataFrame()
    
    # Combine FC data
    all_fc_data = []
    
    if arc_data['fc_data'] is not None:
        arc_fc = arc_data['fc_data'].copy()
        arc_fc['source'] = 'ARC'
        all_fc_data.append(arc_fc)
    
    if osf_data['fc_data'] is not None:
        osf_fc = osf_data['fc_data'].copy()
        osf_fc['source'] = 'OSF'
        all_fc_data.append(osf_fc)
    
    # Combine FC data
    if all_fc_data:
        combined_fc = pd.concat(all_fc_data, axis=0)
    else:
        combined_fc = pd.DataFrame()
    
    # Prepare final dataset structure
    dataset = {
        'metadata': combined_metadata,
        'fc_data': combined_fc,
        'arc_data': arc_data,
        'figshare_data': figshare_data,
        'osf_data': osf_data
    }
    
    return dataset

if __name__ == "__main__":
    # Test the preprocessing pipeline
    dataset = prepare_combined_dataset()
    print("Data sources loaded:")
    print(f"ARC data: {len(dataset['arc_data']['metadata']) if dataset['arc_data']['metadata'] is not None else 0} subjects")
    print(f"FigShare data: {len(dataset['figshare_data']['metadata']) if dataset['figshare_data']['metadata'] is not None else 0} subjects")
    print(f"OSF data: {len(dataset['osf_data']['metadata']) if dataset['osf_data']['metadata'] is not None else 0} subjects")
    
    print("\nCombined dataset:")
    print(f"Metadata: {len(dataset['metadata'])} subjects")
    print(f"FC data: {len(dataset['fc_data'])} subjects")