import numpy as np
import pandas as pd


def fetch_students():
    ''' Fetches the two dataset csv files and merges it '''
    student_mat = pd.read_csv("dataset/student-mat.csv")
    student_por = pd.read_csv("dataset/student-por.csv")
    students = pd.concat([student_mat, student_por])
    return students


def create_data_for_nn(students_dataframe):
    input_data = pd.DataFrame()
    output_data = pd.DataFrame()

    # Input data

    # Numerical
    # Age
    input_data['age'] = pd.Series(
        data=students_dataframe['age'].values, index=students_dataframe.index)
    # Absences count
    input_data['absences'] = pd.Series(
        data=students_dataframe['absences'].values, index=students_dataframe.index)

    # Family relationship status [bad to good -> 0-1]
    input_data['famrel'] = pd.Series(
        data=((students_dataframe['famrel'].values - 1) / 4), index=students_dataframe.index)
    # Health status [bad to good -> 0-1]
    input_data['health'] = pd.Series(
        data=((students_dataframe['health'].values - 1) / 4), index=students_dataframe.index)
    # Free time after school [0-1]
    input_data['freetime'] = pd.Series(
        data=((students_dataframe['freetime'].values - 1) / 4), index=students_dataframe.index)
    # Going out with friends [0-1]
    input_data['goout'] = pd.Series(
        data=((students_dataframe['goout'].values - 1) / 4), index=students_dataframe.index)

    # Travel time in minutes [0 to 60+ minutes -> 0 to 1]
    input_data['traveltime'] = pd.Series(
        data=((students_dataframe['traveltime'].values) / 4), index=students_dataframe.index)
    # Weekly study time in hours [0 to 10+ hours -> 0 to 1]
    input_data['studytime'] = pd.Series(
        data=((students_dataframe['studytime'].values) / 4), index=students_dataframe.index)
    # Number of past class failures [0 to 4+ failures -> 0 to 1]
    input_data['failures'] = pd.Series(
        data=((students_dataframe['failures'].values) / 4), index=students_dataframe.index)

    # School success [Bad to good -> 0-1]
    # Rounded average of the G1, G2 and G3 divided by 20 will be used as school success from 0 to 1
    input_data['success'] = pd.Series(data=(students_dataframe['G1'] +
                                            students_dataframe['G2'] +
                                            students_dataframe['G3']) / 3 / 20,
                                      index=students_dataframe.index)

    # Mother education status
    # [None, Primary education (4th grade), 5th to 9th grade, Secondary education, Higher education]
    input_data['Medu_none'] = pd.Series(data=0, index=students_dataframe.index)
    input_data['Medu_primary'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Medu_fivenine'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Medu_secondary'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Medu_higher'] = pd.Series(
        data=0, index=students_dataframe.index)

    input_data.loc[students_dataframe['Medu'] == 0, 'Medu_none'] = 1
    input_data.loc[students_dataframe['Medu'] == 1, 'Medu_primary'] = 1
    input_data.loc[students_dataframe['Medu'] == 2, 'Medu_fivenine'] = 1
    input_data.loc[students_dataframe['Medu'] == 3, 'Medu_secondary'] = 1
    input_data.loc[students_dataframe['Medu'] == 4, 'Medu_higher'] = 1

    # Father education status
    # [None, Primary education (4th grade), 5th to 9th grade, Secondary education, Higher education]
    input_data['Fedu_none'] = pd.Series(data=0, index=students_dataframe.index)
    input_data['Fedu_primary'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fedu_fivenine'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fedu_secondary'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fedu_higher'] = pd.Series(
        data=0, index=students_dataframe.index)

    input_data.loc[students_dataframe['Fedu'] == 0, 'Fedu_none'] = 1
    input_data.loc[students_dataframe['Fedu'] == 1, 'Fedu_primary'] = 1
    input_data.loc[students_dataframe['Fedu'] == 2, 'Fedu_fivenine'] = 1
    input_data.loc[students_dataframe['Fedu'] == 3, 'Fedu_secondary'] = 1
    input_data.loc[students_dataframe['Fedu'] == 4, 'Fedu_higher'] = 1

    # Mother's job
    # [Teacher, Health care related, Civil services, At home, Other]
    input_data['Mjob_teacher'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Mjob_health'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Mjob_civilser'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Mjob_athome'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Mjob_other'] = pd.Series(
        data=0, index=students_dataframe.index)

    input_data.loc[students_dataframe['Mjob'] == 'teacher', 'Mjob_teacher'] = 1
    input_data.loc[students_dataframe['Mjob'] == 'health', 'Mjob_health'] = 1
    input_data.loc[students_dataframe['Mjob']
                   == 'services', 'Mjob_civilser'] = 1
    input_data.loc[students_dataframe['Mjob'] == 'at_home', 'Mjob_athome'] = 1
    input_data.loc[students_dataframe['Mjob'] == 'other', 'Mjob_other'] = 1

    # Father's job
    # [Teacher, Health care related, Civil services, At home, Other]
    input_data['Fjob_teacher'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fjob_health'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fjob_civilser'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fjob_athome'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['Fjob_other'] = pd.Series(
        data=0, index=students_dataframe.index)

    input_data.loc[students_dataframe['Fjob'] == 'teacher', 'Fjob_teacher'] = 1
    input_data.loc[students_dataframe['Fjob'] == 'health', 'Fjob_health'] = 1
    input_data.loc[students_dataframe['Fjob']
                   == 'services', 'Fjob_civilser'] = 1
    input_data.loc[students_dataframe['Fjob'] == 'at_home', 'Fjob_athome'] = 1
    input_data.loc[students_dataframe['Fjob'] == 'other', 'Fjob_other'] = 1

    # Reason to chose this school
    # [ Close to home, School reputation, Course preference, Other ]
    input_data['reason_closehome'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['reason_rep'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['reason_pref'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data['reason_other'] = pd.Series(
        data=0, index=students_dataframe.index)

    input_data.loc[students_dataframe['reason']
                   == 'home', 'reason_closehome'] = 1
    input_data.loc[students_dataframe['reason']
                   == 'reputation', 'reason_rep'] = 1
    input_data.loc[students_dataframe['reason'] == 'course', 'reason_pref'] = 1
    input_data.loc[students_dataframe['reason'] == 'other', 'reason_other'] = 1

    # One hot
    # Sex [M(Male) = 0, F(Female) = 1]
    input_data['sex'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['sex'] == 'F', 'sex'] = 1
    # Address [R(Rural) = 0, U(Urban) = 1]
    input_data['address'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['address'] == 'U', 'address'] = 1
    # Family size [LE3(Less or equal than 3) = 0, GT3(Greater than 3) = 1]
    input_data['famsize'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['famsize'] == 'GT3', 'famsize'] = 1
    # Parent cohabitation status [T(Together) = 0, A(Apart) = 1]
    input_data['Pstatus'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['Pstatus'] == 'A', 'Pstatus'] = 1
    # Extra educational support [no = 0, yes = 1]
    input_data['schoolsup'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['schoolsup'] == 'yes', 'schoolsup'] = 1
    # Family educational support [no = 0, yes = 1]
    input_data['famsup'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['famsup'] == 'yes', 'famsup'] = 1
    # Extra curricular activites [no = 0, yes = 1]
    input_data['activities'] = pd.Series(
        data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['activities'] == 'yes', 'activities'] = 1
    # Extra curricular activites [no = 0, yes = 1]
    input_data['nursery'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['nursery'] == 'yes', 'nursery'] = 1
    # Wants higher education [no = 0, yes = 1]
    input_data['higher'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['higher'] == 'yes', 'higher'] = 1
    # Internet access at home [no = 0, yes = 1]
    input_data['internet'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['internet'] == 'yes', 'internet'] = 1
    # Has romantic relationship [no = 0, yes = 1]
    input_data['romantic'] = pd.Series(data=0, index=students_dataframe.index)
    input_data.loc[students_dataframe['romantic'] == 'yes', 'romantic'] = 1

    # Output data

    # Alcohol consumption on workdays [0-1]
    output_data['alcohol_work'] = pd.Series(
        data=((students_dataframe['Dalc'].values - 1) / 4), index=students_dataframe.index)

    # Alcohol consumption on workdays [0-1]
    output_data['alcohol_wkend'] = pd.Series(
        data=((students_dataframe['Walc'].values - 1) / 4), index=students_dataframe.index)

    return (input_data, output_data)


def create_input_matrix_from_dataframe(students_df):
    arr = []
    for _, row in students_df.iterrows():
        input_row = create_input_row(
            row['sex'], row['age'], row['address'], row['famsize'], row['Medu'],
            row['Fedu'], row['Mjob'], row['Fjob'], row['reason'],
            row['traveltime'] / 4, row['studytime'] / 4, row['Pstatus'],
            (row['G1'] + row['G2'] + row['G3']) / 3 / 20,
            row['failures'] / 4, row['schoolsup'], row['famsup'], ['activites'],
            row['nursery'], row['higher'], row['internet'], row['romantic'],
            (row['famrel'] - 1) / 4, (row['freetime'] - 1) / 4,
            (row['goout'] - 1) / 4, (row['health'] - 1) / 4, row['absences']
        )
        arr.append(input_row)
    return np.array(arr)


def create_output_matrix_from_dataframe(students_df):
    arr = []
    for _, row in students_df.iterrows():
        input_row = create_output_row(
            (row['Dalc'] - 1) / 4,
            (row['Walc'] - 1) / 4
        )
        arr.append(input_row)
    return np.array(arr)


def create_input_row(sex, age, address, famsize, medu, fedu,
                     mjob, fjob, reason, traveltime, studytime, pstatus,
                     success, failures, schoolsup, famsup, activities,
                     nursery, higher, internet, romantic, famrel, freetime,
                     goout, health, absences):
    row = [
        age,
        absences,

        famrel,
        health,
        freetime,
        goout,
        traveltime,
        studytime,
        failures,
        success,

        1 if medu is 'none' or medu is 0 else 0,
        1 if medu is 'primary' or medu is 1 else 0,
        1 if medu is 'fivenine' or medu is 2 else 0,
        1 if medu is 'secondary' or medu is 3 else 0,
        1 if medu is 'higher' or medu is 4 else 0,

        1 if fedu is 'none' or fedu is 0 else 0,
        1 if fedu is 'primary' or fedu is 1 else 0,
        1 if fedu is 'fivenine' or fedu is 2 else 0,
        1 if fedu is 'secondary' or fedu is 3 else 0,
        1 if fedu is 'higher' or fedu is 4 else 0,

        1 if mjob is 'teacher' else 0,
        1 if mjob is 'health' else 0,
        1 if mjob is 'services' or mjob is 'civilser' else 0,
        1 if mjob is 'athome' or mjob is 'at_home' else 0,
        1 if mjob is 'other' else 0,

        1 if fjob is 'teacher' else 0,
        1 if fjob is 'health' else 0,
        1 if fjob is 'services' or fjob is 'civilser' else 0,
        1 if fjob is 'athome' or fjob is 'at_home' else 0,
        1 if fjob is 'other' else 0,

        1 if reason is 'home' or reason is 'closehome' else 0,
        1 if reason is 'reputation' else 0,
        1 if reason is 'course' or reason is 'pref' else 0,
        1 if reason is 'other' else 0,

        1 if sex is 'F' or sex is 1 or sex is True else 0,
        1 if address is 'U' or address is 1 or address is True else 0,
        1 if famsize is 'GT3' or famsize is 1 or address is True else 0,
        1 if pstatus is 'A' or pstatus is 1 or pstatus is True else 0,
        1 if schoolsup is 'yes' or schoolsup is 1 or schoolsup is True else 0,
        1 if famsup is 'yes' or famsup is 1 or famsup is True else 0,
        1 if activities is 'yes' or activities is 1 or activities is True else 0,
        1 if nursery is 'yes' or nursery is 1 or nursery is True else 0,
        1 if higher is 'yes' or higher is 1 or higher is True else 0,
        1 if internet is 'yes' or internet is 1 or internet is True else 0,
        1 if romantic is 'yes' or romantic is 1 or romantic is True else 0
    ]

    return np.array(row)


def create_output_row(workday_alc, weekend_alc):
    return np.array([
        workday_alc,
        weekend_alc
    ])


def main():
    ''' Created for test purposes '''
    print(create_output_matrix_from_dataframe(fetch_students()))
    pass


if __name__ == '__main__':
    main()
