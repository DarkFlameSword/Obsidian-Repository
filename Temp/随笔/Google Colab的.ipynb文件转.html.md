---
date: 2025-10-24
author:
  - Siyuan Liu
tags:
  - 随笔
---
# Step 1. 下载colab的.ipynb文件
![[Pasted image 20251024154005.png]]
# Step 2. 在google drive新建一个colab
![[Pasted image 20251024153939.png]]
# Step 3. 上传.ipynb文件

# Step 4. 运行代码
```
%%shell

jupyter nbconvert --to html YOUR_DOCUMENT_PATH.ipynb
```


# 报错解决方案
如果出现下面的报错
```
**File "/usr/local/lib/python3.12/dist-packages/nbconvert/filters/widgetsdatatypefilter.py", line 58, in __call__ metadata["widgets"][WIDGET_STATE_MIMETYPE]["state"] ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
^^^^^^^^^ 
KeyError: 'state'**.
```

1. 用notepad打开原始 .ipynb file
2. Copy 整个JSON内容
3. Go to [https://jqplay.org/](https://jqplay.org/)
4. 粘贴内容到左下角`JSON`区域
5. 在左上角`QUERY`中输入 `del(.metadata.widgets)`
6. 复制`OUTPUT`中的结果
7. 替换粘贴到原始 .ipynb
8. 保存notepad文件
9. 重新执行上面的[[Google Colab的.ipynb文件转.html#Step 4. 运行代码]]