import pandas as pd
import numpy as np
from typing import List

path = "./SL3/SL3"

path_misc = "./SL3_miscs/"

total_sub_num = 16

seq_length  = 5


def read_dat_file(path: str):
    data = pd.read_csv(path, delimiter='\t')
    return data


def read_dat_files_subjs_list(subjs_list: List[int]):
    """
    Reads the corresponding dat files of subjects and converts them to a list of dataframes.
    """
    data = [read_dat_file(path + "_s" + f'{sub:02}' + ".dat") for sub in subjs_list]
    for sub in subjs_list:
        data[sub - 1]['SubNum'] = sub
    return data



def remove_no_go_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes no-go trials
    """

    return subj[subj['announce'] == 0]


def select_training_trials(subjs: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the training trials
    """

    return subjs[subjs['trialType'] == 2]



def add_IPI(subj: pd.DataFrame):
    """
    Adds interpress intervals to a subject's dataframe
    """

    for i in range(seq_length-1):
        col1 = 'RT'+str(i+1)
        col2 = 'RT'+str(i+2)
        new_col = 'IPI'+str(i+1)
        subj[new_col] = subj[col2] - subj[col1]

    subj['IPI0'] = subj['RT1']




def finger_melt_IPIs(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each IPI in the whole experiment adding two columns, "IPI_Number" determining the order of IPI
    and "IPI_Value" determining the time of IPI
    """

    
    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'seqType', 'board', 'day', 'trialPoints', 'latePress',
                              'hardPress', 'seqError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('IPI')],
                    var_name='IPI_Number', 
                    value_name='IPI_Value')
    

    subj_melted['N'] = (subj_melted['IPI_Number'].str.extract('(\d+)').astype('int64') + 1)

    

    
    return subj_melted



def finger_melt_responses(subj: pd.DataFrame) -> pd.DataFrame:

    subj_melted = pd.melt(subj, 
                    id_vars=['BN', 'TN', 'SubNum', 'seqType', 'board', 'day', 'trialPoints', 'latePress',
                              'hardPress', 'seqError'], 
                    value_vars =  [_ for _ in subj.columns if _.startswith('resp')],
                    var_name='Response_Number', 
                    value_name='Response_Value')
    
    subj_melted['N'] = subj_melted['Response_Number'].str.extract('(\d+)').astype('int64')

    return subj_melted


def finger_melt(subj: pd.DataFrame) -> pd.DataFrame:
    melt_IPIs = finger_melt_IPIs(subj)
    melt_responses = finger_melt_responses(subj)
    
    merged_df = melt_IPIs.merge(melt_responses, on = ['BN', 'TN', 'SubNum', 'seqType', 
                                                      'board', 'day', 'trialPoints', 
                                                      'latePress','hardPress', 'seqError', 'N'] )

    return merged_df



def remove_error_trials(subj: pd.DataFrame) -> pd.DataFrame:
    """
    Removes error trials from the dat file of a subject
    """

    return subj[(subj['trialPoints'] >= 0)]


def finger_melt_Forces(subjs_force: pd.DataFrame) -> pd.DataFrame:
    """
    Creates seperate row for each Finger Force in the whole experiment adding two columns, "Force_Number" determining the order of Force
    and "Force_Value" determining the time of Force
    """

    
    subj_force_melted = pd.melt(subjs_force, 
                    id_vars=['state', 'timeReal', 'time','BN', 'TN', 'SubNum', 'seqType', 
                                                      'board', 'day', 'trialPoints', 
                                                      'latePress','hardPress', 'seqError'], 
                    value_vars =  [_ for _ in subjs_force.columns if _.startswith('force')],
                    var_name='Force_Number', 
                    value_name='Force_Value')
    
    return subj_force_melted