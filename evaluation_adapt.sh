/bin/bash

python evaluation_adapt.py 
    --gold spider_data/test_gold.sql \
    --pred out/eval/pred_test.sql \
    --etype exec \
    --db spider_data/test_database \
    --table spider_data/test_tables.json \
    --timeout 5