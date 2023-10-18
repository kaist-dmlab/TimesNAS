nohup python -u searcher.py --task_name classification --gpu 3 --population_size 1000 --data_name JapaneseVowels > logs/TimesNAS_classification_JapaneseVowels.out & 
nohup python -u trainer.py --task_name classification --gpu 0 --population_size 1000 --method_name TimesNAS --data_name JapaneseVowels > logs/TimesNAS_classification_JapaneseVowels.out &

nohup python -u searcher.py --task_name anomaly_detection --gpu 4 --population_size 1000 --data_name PSM > logs/TimesNAS_anomaly_detection_PSM.out &
nohup python -u trainer.py --task_name anomaly_detection --gpu 7 --population_size 1000 --method_name TimesNAS --data_name PSM > logs/TimesNAS_anomaly_detection_PSM.out &

nohup python -u searcher.py --task_name short_term_forecast --gpu 0 --population_size 1000 --data_name M4 > logs/TimesNAS_short_forecast_M4.out &
nohup python -u trainer.py --task_name short_term_forecast --gpu 0 --population_size 1000 --method_name TimesNAS --data_name M4 > logs/TimesNAS_short_forecast_M4.out &

nohup python -u searcher.py --task_name long_term_forecast --gpu 1 --population_size 1000 --data_name ETTh1 > logs/TimesNAS_long_forecast_ETTh1.out &
nohup python -u trainer.py --task_name long_term_forecast --gpu 1 --population_size 1000 --method_name TimesNAS --data_name ETTh1 > logs/TimesNAS_long_forecast_ETTh1.out &

nohup python -u searcher.py --task_name imputation --gpu 0 --population_size 1000  --data_name ETTh1 > logs/TimesNAS_imputation_ETTh1.out & 
nohup python -u trainer.py --task_name imputation --gpu 3 --population_size 1000 --method_name TimesNAS --data_name ETTh1 > logs/TimesNAS_imputation_ETTh1.out &