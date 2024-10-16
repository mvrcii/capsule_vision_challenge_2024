import pandas as pd


def extract_dataset_name(path):
    path_parts = path.split('\\') if '\\' in path else path.split('/')
    if len(path_parts) > 3:

        if path_parts[3] == 'KVASIR':
            return 'kvasircapsule'
        elif path_parts[3] == 'SEE-AI':
            return 'seeai'
        elif path_parts[3] == 'AIIMS':
            return 'aiims'
        elif path_parts[3] == 'KID':
            return 'kid'
        else:
            return None
    return None


endoextend_df = pd.read_csv('train_val_endoextend.csv')
cvip_df = pd.read_csv('train_val_cvip.csv')

print("CVIP original size:", len(cvip_df))
print("EndoExtend original size:", len(endoextend_df))

# CLEANING
cvip_df = cvip_df[cvip_df['dataset'] != 'aiims']
print("CVIP removed AIIMS size:", len(cvip_df))
endoextend_df = endoextend_df[(endoextend_df['dataset'] == 'kid') | (endoextend_df['dataset'] == 'seeai') | (endoextend_df['dataset'] == 'kvasircapsule')]
print("Total (seeai, kid, kvasir):", len(endoextend_df), len(cvip_df))

endoextend_filenames = endoextend_df['proposed_name'].unique()
cvip_filenames = cvip_df['proposed_name'].unique()
matching_filenames = list(set(endoextend_filenames).intersection(set(cvip_filenames)))
matching_filenames = list(matching_filenames)
print("Overlapping Samples:", len(matching_filenames))

endoextend_train_df = endoextend_df[endoextend_df['fold'] != 1]
cvip_val_df = cvip_df[cvip_df['fold'] == 1]
print("Train vs. Val Overlap:", len(endoextend_train_df), len(cvip_val_df))


endoextend_train_filenames = endoextend_train_df['proposed_name'].unique()
cvip_val_filenames = cvip_val_df['proposed_name'].unique()
matching_overlap_filenames = list(set(endoextend_train_filenames).intersection(set(cvip_val_filenames)))
matching_overlap_filenames = list(matching_overlap_filenames)
print("Overlapping Samples that are relevant:", len(matching_overlap_filenames))



# sub_endoextend_df = endoextend_df[endoextend_df['proposed_name'].isin(matching_filenames)]
#
# print(sub_endoextend_df.value_counts('dataset'))
