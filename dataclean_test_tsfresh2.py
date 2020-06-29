import pandas as pd
import glob
import natsort
import csv
import os
from pathlib import Path
import numpy as np
import re
import tsfresh
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.distribution import MultiprocessingDistributor

path = r'/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/'
all_files = glob.glob(path + "/raw_testx/*.m**")
all_files = natsort.natsorted(all_files, reverse=False)

labels_df = pd.read_csv(path + 'LABEL_Testx.csv', index_col=None, header=0)

extracted_eeg = []

labels_anxiety = []
#labels_apathy = []
#labels_cognition = []
#labels_depression = []

for filename in all_files:

    patient_id = os.path.basename(filename)
    patient_id = os.path.splitext(patient_id)[0]

    f = open(filename, "r")
    lines = f.read().splitlines()
    f.close()

    lines.pop(0)

    lines_csv = []

    for line in lines:

        line = line.strip()
        line = re.sub(' +', ',', line)

        lines_csv.append(line)

    f_temp = os.path.join(os.path.dirname(os.path.abspath(filename)), 'tmp', os.path.basename(filename) + '.csv')

    f = open(f_temp, 'w')
    f_contents = '\n'.join(lines_csv)
    f.write(f_contents)
    f.close()
    for csv in all_files:
        df = pd.read_csv(f_temp, index_col=None, header=0)
        df['ID'] = os.path.basename(csv)
        df['ID'] = df['ID'].astype(str).str[:-4]
        df = df[['ID', 'Fp2-Cz', 'Fp1-Cz', 'F8-Cz', 'F7-Cz', 'F4-Cz', 'F3-Cz', 'A2-Cz', 'A1-Cz' , 'T4-Cz', 'T3-Cz', 'C4-Cz' , 'C3-Cz', 'T6-Cz', 'T5-Cz' ,'P4-Cz' ,'P3-Cz','O2-Cz','O1-Cz','Fz-Cz','Cz-Cz','Pz-Cz']]
        df.append(df)
        #print(df)
    diagnosis = labels_df.loc[labels_df['ID'] == int(patient_id)]

    list_anxiety = []
    #list_depression = []
    #list_cognition = []
    #list_apathy = []

    list_anxiety += len(df.index) * [diagnosis.iloc[0]['Anxiety']]
    #list_depression += len(df.index) * [diagnosis.iloc[0]['Depression']]
    #list_cognition += len(df.index) * [diagnosis.iloc[0]['COG']]
    #list_apathy += len(df.index) * [diagnosis.iloc[0]['Apathy']]

    labels_anxiety.append(pd.DataFrame(list_anxiety))
    #labels_depression.append(pd.DataFrame(list_depression))
    #labels_cognition.append(pd.DataFrame(list_cognition))
    #labels_apathy.append(pd.DataFrame(list_apathy))

    extracted_eeg.append(df)

#new_row = pd.DataFrame({'Fp2-Cz', 'Fp1-Cz', 'F8-Cz', 'F7-Cz', 'F4-Cz', 'F3-Cz', 'A2-Cz', 'A1-Cz' , 'T4-Cz', 'T3-Cz', 'C4-Cz' , 'C3-Cz', 'T6-Cz', 'T5-Cz' ,'P4-Cz' ,'P3-Cz','O2-Cz','O1-Cz','Fz-Cz','Cz-Cz' ,'Pz-Cz'},
#                                                            index =[0])

labels_anxiety_df = pd.concat(labels_anxiety, axis=0, ignore_index=True)
#labels_apathy_df = pd.concat(labels_apathy, axis=0, ignore_index=True)
#labels_cognition_df = pd.concat(labels_cognition, axis=0, ignore_index=True)
#labels_depression_df = pd.concat(labels_depression, axis=0, ignore_index=True)

extracted_eeg_df = pd.concat(extracted_eeg, axis=0, ignore_index=True)
#print(extracted_eeg_df)

time = extracted_eeg_df.index.values
#extracted_eeg_df.insert(0,column="ID",value=ID)
extracted_eeg_df.insert(0,column="time",value=time)
extracted_eeg_df = extracted_eeg_df[['ID','time', 'Fp2-Cz', 'Fp1-Cz', 'F8-Cz', 'F7-Cz', 'F4-Cz', 'F3-Cz', 'A2-Cz', 'A1-Cz' , 'T4-Cz', 'T3-Cz', 'C4-Cz' , 'C3-Cz', 'T6-Cz', 'T5-Cz' ,'P4-Cz' ,'P3-Cz','O2-Cz','O1-Cz','Fz-Cz','Cz-Cz','Pz-Cz']]
#del extracted_eeg_df['F8-Fz']
#del extracted_eeg_df['Fz-F7']
#del extracted_eeg_df['F8-F7']
#del extracted_eeg_df['C4-Fp2']
#del extracted_eeg_df['C3-Fp1']

# new line
#label = pd.DataFrame({'label'}, index=[0])
# concatenate two dataframe
#label_depression_df = pd.concat([label,labels_depression_df.ix[:]).reset_index(drop = True)
#labels_depression_df.loc[-1] = ['label']  # adding a row
#labels_depression_df.index = labels_depression_df.index + 1  # shifting index
#labels_depression_df = labels_depression_df.sort_index()  # sorting by index

print (extracted_eeg_df.head(5))
labels_anxiety_df.columns = ['label']
#labels_depression_df.columns = ['label']
#labels_cognition_df.columns = ['label']
#labels_apathy_df.columns = ['label']

#extracted_eeg_ts_df = tsfresh.utilities.dataframe_functions.roll_time_series(extracted_eeg_df,column_id="ID")

#extracted_features_eeg = extract_features(extracted_eeg_df, column_id='ID', column_sort='time')
#impute(extracted_features_eeg)
#features_filtered = select_features(extracted_features_eeg)

#Distributor = MultiprocessingDistributor(n_workers=6,
#                                         disable_progressbar=False,
#                                         progressbar_title="Feature Extraction")

#features_filtered_direct_eeg = extract_relevant_features(extracted_eeg_df, extracted_eeg_df_y,column_id='ID', column_sort='time')
extracted_features_eeg = extract_features(timeseries_container=extracted_eeg_df,column_id='ID', column_sort='time',n_jobs=14)

#features_filtered_direct_eeg = extract_relevant_features(timeseries_container=extracted_eeg_df,y=
#column_id='ID', column_sort='time',n_jobs=4)




extracted_features_eeg.to_csv('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_testx2_ts_extracted.csv')
labels_anxiety_df.to_csv(
        '/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_testx2_label_ts_extracted.csv')
#labels_depression_df.to_csv('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_3_label_d_extracted.csv')
#labels_cognition_df.to_csv('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_3_label_c_extracted.csv')
#labels_apathy_df.to_csv('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_3_label_a_extracted.csv')

extracted_features_eeg.to_pickle('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_ts_extracted.pkl')
labels_anxiety_df.to_pickle(
        '/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_testx2_label_ts_extracted.pkl')
#labels_cognition_df.to_pickle('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_3_label_c_extracted.pkl')
#labels_apathy_df.to_pickle('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_3_label_a_extracted.pkl')
#labels_depression_df.to_pickle('/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_3_label_d_extracted.pkl')
#labels_cognition_df = pd.read_csv(
#    '/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_2_label_c_extracted.csv',
#    index_col=None, header=0)
#labels_apathy_df = pd.read_csv(
#    '/home/forensic2/PycharmProjects/Thesis_test/EEG_package_Regression/raw/output/eeg_2_label_a_extracted.csv',
#    index_col=None, header=0)


