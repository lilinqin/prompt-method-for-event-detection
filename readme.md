基于prompt方法的事件检测
=======================
1.语料
-------
ACE2005中文事件数据  

2.环境
------
具体查看requirements.txt  

3.运行
------
run.sh中包含了7类方法，区分在于是否使用了prompt方法，以及是连续prompt还是提示prompt，
同时也涉及了是否使用MLM  

运行代码可以注释掉run.sh中的部分内容，然后运行下面的脚本
```Bash
sh run.sh
```

或者可以根据自己的需求写脚本运行
```Bash
python prompt_tuning.py \
    --seed $seed \
    --manual_template [MASK]事件： \
    --model manual_prompt \
    --mask_id  1 \
    --epochs 10 \
    --fine_tuning \
    --lr 2e-5
```

