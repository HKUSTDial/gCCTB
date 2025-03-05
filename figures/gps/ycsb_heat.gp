# set terminal pngcairo size 1800,750 enhanced font 'Verdana,12'   ## 格式，大小和字体

set terminal pdf size 18,7.5 enhanced font 'Times-New-Roman,24,Bold'

set output "../pdf/ycsb_heat_ro.pdf"  ###输出的文件名
unset key
set tics nomirror out scale 0.75
set cblabel "Throughput(10^6 txn/s)" offset 0,0
#set cbtics format "%.1e"
set title offset 0,-0.5

set tmargin 2  # 设置图表的上边距
set bmargin -2  # 设置图表的下边距
set lmargin 4  # 设置图表的左边距
set rmargin 7  # 设置图表的右边距

set xrange [-0.5:31.5]
set yrange [-0.5:5.5]
set xlabel 'Block Size' offset 0,0.8
set ylabel 'Warp Density' offset 1.4,0
set mxtics 2
set mytics 4
set xtics offset 0,0.3
set ytics offset 0.6,0
set datafile separator ","


###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 2,4  margins 0.1,0.8,0.1,0.9
set cbrange [10:50] 

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GaccO"
plot '../matrices/gacco_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "Silo"
plot '../matrices/silo_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TicToc"
plot '../matrices/tictoc_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GPUTx"
plot '../matrices/gputx_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_nw"
plot '../matrices/tpl_nw_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_wd"
plot '../matrices/tpl_wd_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TO"
plot '../matrices/to_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "MVCC"
plot '../matrices/mvcc_ro.txt' matrix rowheaders columnheaders with image

unset cbrange
unset multiplot  ###退出多图模式，完成绘图并保存

###########################################################################################

set output "../pdf/ycsb_heat_mc.pdf"  ###输出的文件名

###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 2,4 

set cbrange [4.0:26] 

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GaccO"
plot '../matrices/gacco_mc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "Silo"
plot '../matrices/silo_mc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TicToc"
plot '../matrices/tictoc_mc.txt' matrix rowheaders columnheaders with image

set cbrange [3.2:8.0] 

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GPUTx"
plot '../matrices/gputx_mc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_nw"
plot '../matrices/tpl_nw_mc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_wd"
plot '../matrices/tpl_wd_mc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TO"
plot '../matrices/to_mc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "MVCC"
plot '../matrices/mvcc_mc.txt' matrix rowheaders columnheaders with image

unset cbrange
unset multiplot  ###退出多图模式，完成绘图并保存

###########################################################################################

set output "../pdf/ycsb_heat_hc.pdf"  ###输出的文件名


###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 2,4 

set cbrange [0:3.5] 

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GaccO"
plot '../matrices/gacco_hc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "Silo"
plot '../matrices/silo_hc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TicToc"
plot '../matrices/tictoc_hc.txt' matrix rowheaders columnheaders with image

set cbrange [0.02:0.2]

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GPUTx"
plot '../matrices/gputx_hc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_nw"
plot '../matrices/tpl_nw_hc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_wd"
plot '../matrices/tpl_wd_hc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TO"
plot '../matrices/to_hc.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "MVCC"
plot '../matrices/mvcc_hc.txt' matrix rowheaders columnheaders with image

unset cbrange
unset multiplot  ###退出多图模式，完成绘图并保存