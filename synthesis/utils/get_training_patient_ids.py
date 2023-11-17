
# Purpose: To get the patient ids of the training set from the csv file
import csv
import os

# global variables

### Local
PREFIX_PATH = os.path.join('PATH_TO_DATA', '')
PREFIX_OUTPUT_PATH = PREFIX_PATH

### Server
# PREFIX_PATH = os.path.join('PATH_TO_DATA', '')
# PREFIX_OUTPUT_PATH =  PREFIX_PATH

CSV_PATH = os.path.join(PREFIX_PATH, 'Clinical_and_Other_Features.csv')

OUTPUT_FILE_TYPE = '.txt'
OUTPUT_FILE_PATH = os.path.join(PREFIX_OUTPUT_PATH, f'patient_ids_without_seg_mask{OUTPUT_FILE_TYPE}')

TESTSET_LIST = ['Breast_MRI_001', 'Breast_MRI_002', 'Breast_MRI_005', 'Breast_MRI_009', 'Breast_MRI_010',
                'Breast_MRI_012', 'Breast_MRI_019', 'Breast_MRI_021', 'Breast_MRI_022', 'Breast_MRI_028',
                'Breast_MRI_032', 'Breast_MRI_041', 'Breast_MRI_043', 'Breast_MRI_044', 'Breast_MRI_045',
                'Breast_MRI_048', 'Breast_MRI_051', 'Breast_MRI_055', 'Breast_MRI_057', 'Breast_MRI_059',
                'Breast_MRI_060', 'Breast_MRI_061', 'Breast_MRI_064', 'Breast_MRI_069', 'Breast_MRI_071',
                'Breast_MRI_077', 'Breast_MRI_082', 'Breast_MRI_091', 'Breast_MRI_097', 'Breast_MRI_099',
                'Breast_MRI_101', 'Breast_MRI_103', 'Breast_MRI_104', 'Breast_MRI_105', 'Breast_MRI_107',
                'Breast_MRI_114', 'Breast_MRI_115', 'Breast_MRI_116', 'Breast_MRI_117', 'Breast_MRI_119',
                'Breast_MRI_120', 'Breast_MRI_123', 'Breast_MRI_129', 'Breast_MRI_132', 'Breast_MRI_134',
                'Breast_MRI_136', 'Breast_MRI_137', 'Breast_MRI_141', 'Breast_MRI_142', 'Breast_MRI_144',
                'Breast_MRI_150', 'Breast_MRI_156', 'Breast_MRI_157', 'Breast_MRI_160', 'Breast_MRI_163',
                'Breast_MRI_167', 'Breast_MRI_168', 'Breast_MRI_176', 'Breast_MRI_177', 'Breast_MRI_178',
                'Breast_MRI_180', 'Breast_MRI_183', 'Breast_MRI_185', 'Breast_MRI_189', 'Breast_MRI_192',
                'Breast_MRI_198', 'Breast_MRI_202', 'Breast_MRI_205', 'Breast_MRI_211', 'Breast_MRI_218',
                'Breast_MRI_225', 'Breast_MRI_228', 'Breast_MRI_233', 'Breast_MRI_234', 'Breast_MRI_236',
                'Breast_MRI_237', 'Breast_MRI_239', 'Breast_MRI_240', 'Breast_MRI_244', 'Breast_MRI_253',
                'Breast_MRI_255', 'Breast_MRI_258', 'Breast_MRI_260', 'Breast_MRI_265', 'Breast_MRI_268',
                'Breast_MRI_269', 'Breast_MRI_271', 'Breast_MRI_275', 'Breast_MRI_282', 'Breast_MRI_283',
                'Breast_MRI_287', 'Breast_MRI_290', 'Breast_MRI_298', 'Breast_MRI_301', 'Breast_MRI_303',
                'Breast_MRI_304', 'Breast_MRI_306', 'Breast_MRI_307', 'Breast_MRI_313', 'Breast_MRI_317',
                'Breast_MRI_323', 'Breast_MRI_328', 'Breast_MRI_333', 'Breast_MRI_338', 'Breast_MRI_345',
                'Breast_MRI_350', 'Breast_MRI_353', 'Breast_MRI_356', 'Breast_MRI_360', 'Breast_MRI_368',
                'Breast_MRI_377', 'Breast_MRI_378', 'Breast_MRI_383', 'Breast_MRI_386', 'Breast_MRI_387',
                'Breast_MRI_395', 'Breast_MRI_397', 'Breast_MRI_398', 'Breast_MRI_399', 'Breast_MRI_400',
                'Breast_MRI_407', 'Breast_MRI_408', 'Breast_MRI_412', 'Breast_MRI_414', 'Breast_MRI_424',
                'Breast_MRI_428', 'Breast_MRI_429', 'Breast_MRI_431', 'Breast_MRI_435', 'Breast_MRI_438',
                'Breast_MRI_441', 'Breast_MRI_444', 'Breast_MRI_454', 'Breast_MRI_457', 'Breast_MRI_464',
                'Breast_MRI_465', 'Breast_MRI_468', 'Breast_MRI_474', 'Breast_MRI_486', 'Breast_MRI_489',
                'Breast_MRI_491', 'Breast_MRI_501', 'Breast_MRI_506', 'Breast_MRI_507', 'Breast_MRI_508',
                'Breast_MRI_512', 'Breast_MRI_514', 'Breast_MRI_521', 'Breast_MRI_525', 'Breast_MRI_530',
                'Breast_MRI_534', 'Breast_MRI_539', 'Breast_MRI_541', 'Breast_MRI_543', 'Breast_MRI_546',
                'Breast_MRI_552', 'Breast_MRI_558', 'Breast_MRI_559', 'Breast_MRI_560', 'Breast_MRI_562',
                'Breast_MRI_567', 'Breast_MRI_568', 'Breast_MRI_577', 'Breast_MRI_585', 'Breast_MRI_590',
                'Breast_MRI_595', 'Breast_MRI_597', 'Breast_MRI_605', 'Breast_MRI_607', 'Breast_MRI_609',
                'Breast_MRI_610', 'Breast_MRI_612', 'Breast_MRI_614', 'Breast_MRI_615', 'Breast_MRI_616',
                'Breast_MRI_618', 'Breast_MRI_623', 'Breast_MRI_633', 'Breast_MRI_636', 'Breast_MRI_640',
                'Breast_MRI_641', 'Breast_MRI_642', 'Breast_MRI_645', 'Breast_MRI_650', 'Breast_MRI_651',
                'Breast_MRI_652', 'Breast_MRI_656', 'Breast_MRI_660', 'Breast_MRI_662', 'Breast_MRI_663',
                'Breast_MRI_666', 'Breast_MRI_670', 'Breast_MRI_672', 'Breast_MRI_677', 'Breast_MRI_679',
                'Breast_MRI_684', 'Breast_MRI_686', 'Breast_MRI_687', 'Breast_MRI_691', 'Breast_MRI_693',
                'Breast_MRI_694', 'Breast_MRI_697', 'Breast_MRI_718', 'Breast_MRI_724', 'Breast_MRI_725',
                'Breast_MRI_735', 'Breast_MRI_746', 'Breast_MRI_751', 'Breast_MRI_754', 'Breast_MRI_757',
                'Breast_MRI_758', 'Breast_MRI_762', 'Breast_MRI_765', 'Breast_MRI_774', 'Breast_MRI_775',
                'Breast_MRI_778', 'Breast_MRI_780', 'Breast_MRI_789', 'Breast_MRI_790', 'Breast_MRI_792',
                'Breast_MRI_797', 'Breast_MRI_799', 'Breast_MRI_804', 'Breast_MRI_805', 'Breast_MRI_809',
                'Breast_MRI_812', 'Breast_MRI_816', 'Breast_MRI_830', 'Breast_MRI_831', 'Breast_MRI_832',
                'Breast_MRI_833', 'Breast_MRI_834', 'Breast_MRI_836', 'Breast_MRI_839', 'Breast_MRI_847',
                'Breast_MRI_850', 'Breast_MRI_860', 'Breast_MRI_865', 'Breast_MRI_869', 'Breast_MRI_873',
                'Breast_MRI_874', 'Breast_MRI_879', 'Breast_MRI_882', 'Breast_MRI_883', 'Breast_MRI_884',
                'Breast_MRI_885', 'Breast_MRI_886', 'Breast_MRI_891', 'Breast_MRI_899', 'Breast_MRI_907',
                'Breast_MRI_914', 'Breast_MRI_915', 'Breast_MRI_916', 'Breast_MRI_917']


def get_training_set_ids(store_path=None, delimiter='\n', required_prefix= None):
    # with open('PATH_TO_DATA/Duke_Breast_MRI_all_phases.csv') as file_obj:
    training_set_ids = []
    with open(CSV_PATH) as file_obj:
        reader_obj = csv.reader(file_obj, delimiter=';')
        for row in reader_obj:
            patient_id = row[0]
            if patient_id not in TESTSET_LIST:
                if required_prefix is not None:
                    if patient_id.startswith(required_prefix):
                        training_set_ids.append(patient_id)
                else:
                    training_set_ids.append(patient_id)
    if store_path is not None:
        with open(store_path, 'w') as file_obj:
            for patient_id in training_set_ids:
                file_obj.write(patient_id + delimiter)
    return training_set_ids


get_training_set_ids(store_path=OUTPUT_FILE_PATH, delimiter=',', required_prefix='Breast_MRI_')
