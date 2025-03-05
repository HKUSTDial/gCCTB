# set terminal pngcairo size 1200,500 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体
set terminal pdf size 14,5 enhanced font 'Times-New-Roman,36,Bold' 

set output "../pdf/ycsb_ro_diffbs.pdf"  ###输出的文件名

set xlabel "Block Size"     # X轴标签
set xlabel offset 0,0.8 font 'Times-New-Roman,44,Bold'
set ylabel "Throughput(10^6 txn/s)" font 'Times-New-Roman,44,Bold' offset 0,-0.5     # Y轴标签
set ytics font 'Times-New-Roman,44,Bold' 
set grid ytics linewidth 4
set xtics font 'Times-New-Roman,44,Bold' offset 0,0.3
set xtics 2.0

set key font "Times-New-Roman,36,Bold"
set key under box center 
set key offset 0,-0.25
set key samplen 3 

throu_pth = "../mid/ycsb_diffbs_ro_throughput.txt"

set style line 1 lc rgb "#E41A1C" lw 4 dt solid    pt 7  ps 1.5   # 红色 实线 圆点
set style line 2 lc rgb "#377EB8" lw 4 dt solid   pt 5  ps 1.5   # 蓝色 虚线 正方形
set style line 3 lc rgb "#4DAF4A" lw 4 dt solid pt 9  ps 1.5  # 绿色 点划线 三角形
set style line 4 lc rgb "#FF7F00" lw 4 dt solid   pt 11 ps 1.5   # 橙色 点线 菱形
set style line 5 lc rgb "#984EA3" lw 4 dt solid  pt 13 ps 1.5   # 紫色 长虚线 星号
set style line 6 lc rgb "#00CED1" lw 4 dt solid pt 8 ps 1.5  # 青色 短点线 叉号
set style line 7 lc rgb "#666666" lw 4 dt solid pt 6 ps 1.5 # 深灰色 长点线 加号
set style line 8 lc rgb "#F781BF" lw 4 dt solid   pt 4  ps 1.5   # 粉色 短划线 倒三角形

plot throu_pth using 1:(($2)/1e6) title "GaccO" with linespoints ls 1, \
     throu_pth using 1:(($3)/1e6) title "GPUTx" with linespoints ls 2, \
     throu_pth using 1:(($4)/1e6) title "tpl\\\_nw" with linespoints ls 3, \
     throu_pth using 1:(($5)/1e6) title "tpl\\\_wd" with linespoints ls 4, \
     throu_pth using 1:(($6)/1e6) title "TO" with linespoints ls 5, \
     throu_pth using 1:(($7)/1e6) title "MVCC" with linespoints ls 6, \
     throu_pth using 1:(($8)/1e6) title "Silo" with linespoints ls 7, \
     throu_pth using 1:(($9)/1e6) title "TicToc" with linespoints ls 8

set output "../pdf/ycsb_mc_diffbs.pdf"  ###输出的文件名
throu_pth = "../mid/ycsb_diffbs_mc_throughput.txt"

plot throu_pth using 1:(($2)/1e6) title "GaccO" with linespoints ls 1, \
     throu_pth using 1:(($3)/1e6) title "GPUTx" with linespoints ls 2, \
     throu_pth using 1:(($4)/1e6) title "tpl\\\_nw" with linespoints ls 3, \
     throu_pth using 1:(($5)/1e6) title "tpl\\\_wd" with linespoints ls 4, \
     throu_pth using 1:(($6)/1e6) title "TO" with linespoints ls 5, \
     throu_pth using 1:(($7)/1e6) title "MVCC" with linespoints ls 6, \
     throu_pth using 1:(($8)/1e6) title "Silo" with linespoints ls 7, \
     throu_pth using 1:(($9)/1e6) title "TicToc" with linespoints ls 8

set output "../pdf/ycsb_hc_diffbs.pdf"  ###输出的文件名
throu_pth = "../mid/ycsb_diffbs_hc_throughput.txt"

plot throu_pth using 1:(($2)/1e6) title "GaccO" with linespoints ls 1, \
     throu_pth using 1:(($3)/1e6) title "GPUTx" with linespoints ls 2, \
     throu_pth using 1:(($4)/1e6) title "tpl\\\_nw" with linespoints ls 3, \
     throu_pth using 1:(($5)/1e6) title "tpl\\\_wd" with linespoints ls 4, \
     throu_pth using 1:(($6)/1e6) title "TO" with linespoints ls 5, \
     throu_pth using 1:(($7)/1e6) title "MVCC" with linespoints ls 6, \
     throu_pth using 1:(($8)/1e6) title "Silo" with linespoints ls 7, \
     throu_pth using 1:(($9)/1e6) title "TicToc" with linespoints ls 8

####################################################################################

set format y "%.2tx10^{%T}"
set logscale y 2
#set yrange [0.0156:128]
set ylabel "Abort Rate" offset -0.8,0     # Y轴标签
set ytics offset 0.3,0
set key offset -3,-0.25

set output "../pdf/ycsb_mc_diffbs_abr.pdf"  ###输出的文件名
abr_pth = "../mid/ycsb_diffbs_mc_abr.txt"

plot abr_pth using 1:2 title "tpl\\\_nw" with linespoints ls 3, \
     abr_pth using 1:3 title "tpl\\\_wd" with linespoints ls 4, \
     abr_pth using 1:4 title "TO" with linespoints ls 5, \
     abr_pth using 1:5 title "MVCC" with linespoints ls 6, \
     abr_pth using 1:6 title "Silo" with linespoints ls 7, \
     abr_pth using 1:7 title "TicToc" with linespoints ls 8

set output "../pdf/ycsb_hc_diffbs_abr.pdf"  ###输出的文件名
abr_pth = "../mid/ycsb_diffbs_hc_abr.txt"

plot abr_pth using 1:2 title "tpl\\\_nw" with linespoints ls 3, \
     abr_pth using 1:3 title "tpl\\\_wd" with linespoints ls 4, \
     abr_pth using 1:4 title "TO" with linespoints ls 5, \
     abr_pth using 1:5 title "MVCC" with linespoints ls 6, \
     abr_pth using 1:6 title "Silo" with linespoints ls 7, \
     abr_pth using 1:7 title "TicToc" with linespoints ls 8
