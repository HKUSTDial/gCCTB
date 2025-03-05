# set terminal pngcairo size 1800,750 enhanced font 'Verdana,12'   ## 格式，大小和字体

set terminal pdf size 15.5,7.5 enhanced font 'Times-New-Roman,40'

set output "../pdf/ycsb_heat_ro.pdf"  ###输出的文件名
unset key
set tics nomirror out scale 0.75

set title offset 0,-0.5

set tmargin 2  # 设置图表的上边距
set bmargin -2  # 设置图表的下边距
#set lmargin 4  # 设置图表的左边距
#set rmargin 7  # 设置图表的右边距

set xrange [-0.5:31.5]
set yrange [-0.5:5.5]
set xlabel 'Block Size' offset 0,1.4
set ylabel 'Warp Density' offset 1.4,0
set mxtics 2
set mytics 4
set xtics offset 0,0.7 font 'Times-New-Roman,36,Bold'
set ytics offset 0.8,0
set datafile separator ","


###因为是要一张图里4个子图，所以启用了多图模式：
set multiplot layout 2,4  

set colorbox user size 0.015,0.8 origin 0.91,0.13
set cbrange [10:50] 
set cblabel "Throughput(10^6 txn/s)" font 'Times-New-Roman,36' offset -0.9,0
set cbtics offset -1.2,0 font 'Times-New-Roman,36'


########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GaccO" offset 0,-0.9
set size 0.278,0.57
set origin 0,0.485
plot '../matrices/gacco_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "Silo"
set size 0.278,0.57
set origin 0.218,0.485
plot '../matrices/silo_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TicToc"
set size 0.278,0.57
set origin 0.436,0.485
plot '../matrices/tictoc_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "GPUTx"
set size 0.278,0.57
set origin 0.654,0.485
plot '../matrices/gputx_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_nw"
set size 0.278,0.57
set origin 0,0
plot '../matrices/tpl_nw_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "tpl\\\_wd"
set size 0.278,0.57
set origin 0.218,0
plot '../matrices/tpl_wd_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "TO"
set size 0.278,0.57
set origin 0.436,0
plot '../matrices/to_ro.txt' matrix rowheaders columnheaders with image

########## 子图 (1): 绘制函数，设置基本的元素如：标题、坐标范围、图例等
set title "MVCC"
set size 0.278,0.57
set origin 0.654,0
plot '../matrices/mvcc_ro.txt' matrix rowheaders columnheaders with image

unset cbrange
unset multiplot  ###退出多图模式，完成绘图并保存
