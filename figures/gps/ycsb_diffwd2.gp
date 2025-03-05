# set terminal pngcairo size 1200,500 enhanced font 'Roman,20,Bold'   ## 格式，大小和字体
set terminal pdf size 15,5 enhanced font 'Times-New-Roman,44,Bold' 


set style line 1 lc rgb "#E41A1C" lw 4 dt solid    pt 7  ps 1.5   # 红色 实线 圆点
set style line 2 lc rgb "#377EB8" lw 4 dt solid   pt 5  ps 1.5   # 蓝色 虚线 正方形
set style line 3 lc rgb "#4DAF4A" lw 4 dt solid pt 9  ps 1.5  # 绿色 点划线 三角形
set style line 4 lc rgb "#FF7F00" lw 4 dt solid   pt 11 ps 1.5   # 橙色 点线 菱形
set style line 5 lc rgb "#984EA3" lw 4 dt solid  pt 13 ps 1.5   # 紫色 长虚线 星号
set style line 6 lc rgb "#00CED1" lw 4 dt solid pt 8 ps 1.5  # 青色 短点线 叉号
set style line 7 lc rgb "#666666" lw 4 dt solid pt 6 ps 1.5 # 深灰色 长点线 加号
set style line 8 lc rgb "#F781BF" lw 4 dt solid   pt 4  ps 1.5   # 粉色 短划线 倒三角形

set xlabel "Warp Density"      # X轴标签
set xlabel offset 0,0.8 font 'Times-New-Roman,48,Bold'
set xtics font 'Times-New-Roman,37,Bold' offset 0,0.5

set output "../pdf/ycsb_mc_diffwd2.pdf"  ###输出的文件名

set multiplot

set ylabel "Throughput(10^6 txn/s)" font 'Times-New-Roman,48,Bold' offset 1.4,-0.5     # Y轴标签
set ytics font 'Times-New-Roman,44,Bold' 
set grid ytics linewidth 4

set key font "Times-New-Roman,36,Bold"
set key at graph 2.35,screen 0.1
set key box
set key horizontal maxrows 1 columns 8
set key samplen 1
#set key width 0.01

throu_pth = "../mid/ycsb_diffwd_mc_throughput.txt"

set size 0.52,1
set origin 0,0.1
set title '(a) Total Throughput' offset 0, -1
set xlabel 'Warp Density' offset 0,1.1
plot throu_pth using 1:(($2)/1e6) title "GaccO" with linespoints ls 1, \
     throu_pth using 1:(($3)/1e6) title "GPUTx" with linespoints ls 2, \
     throu_pth using 1:(($4)/1e6) title "tpl\\\_nw" with linespoints ls 3, \
     throu_pth using 1:(($5)/1e6) title "tpl\\\_wd" with linespoints ls 4, \
     throu_pth using 1:(($6)/1e6) title "TO" with linespoints ls 5, \
     throu_pth using 1:(($7)/1e6) title "MVCC" with linespoints ls 6, \
     throu_pth using 1:(($8)/1e6) title "Silo" with linespoints ls 7, \
     throu_pth using 1:(($9)/1e6) title "TicToc" with linespoints ls 8
    
unset key

set format y "%.2tx10^{%T}"
set logscale y 2
#set yrange [0.0156:128]
set ylabel "Abort Rate" offset 2.2,0     # Y轴标签
set ytics offset 0.3,0

abr_pth = "../mid/ycsb_diffwd_mc_abr.txt"

set size 0.58,1
set origin 0.445,0.1
set title '(b) Abort Rate'
set xlabel 'Warp Density'
plot abr_pth using 1:2 title "tpl\\\_nw" with linespoints ls 3, \
     abr_pth using 1:3 title "tpl\\\_wd" with linespoints ls 4, \
     abr_pth using 1:4 title "TO" with linespoints ls 5, \
     abr_pth using 1:5 title "MVCC" with linespoints ls 6, \
     abr_pth using 1:6 title "Silo" with linespoints ls 7, \
     abr_pth using 1:7 title "TicToc" with linespoints ls 8

unset multiplot
unset key
unset logscale
unset format y


#####################################################################################

set output "../pdf/ycsb_hc_diffwd2.pdf" ###输出的文件名

set multiplot

set ylabel "Throughput(10^6 txn/s)" font 'Times-New-Roman,48,Bold' offset 2,-0.5     # Y轴标签
set ytics font 'Times-New-Roman,44,Bold' 
set grid ytics linewidth 4

set key font "Times-New-Roman,36,Bold"
set key at graph 2.35,screen 0.1
set key box
set key horizontal maxrows 1 columns 8
set key samplen 1
#set key width 0.01

throu_pth = "../mid/ycsb_diffwd_hc_throughput.txt"

set size 0.52,1
set origin 0,0.1
set title '(a) Total Throughput'
set xlabel 'Warp Density' offset 0,1.1
plot throu_pth using 1:(($2)/1e6) title "GaccO" with linespoints ls 1, \
     throu_pth using 1:(($3)/1e6) title "GPUTx" with linespoints ls 2, \
     throu_pth using 1:(($4)/1e6) title "tpl\\\_nw" with linespoints ls 3, \
     throu_pth using 1:(($5)/1e6) title "tpl\\\_wd" with linespoints ls 4, \
     throu_pth using 1:(($6)/1e6) title "TO" with linespoints ls 5, \
     throu_pth using 1:(($7)/1e6) title "MVCC" with linespoints ls 6, \
     throu_pth using 1:(($8)/1e6) title "Silo" with linespoints ls 7, \
     throu_pth using 1:(($9)/1e6) title "TicToc" with linespoints ls 8
    
unset key

set format y "%.2tx10^{%T}"
set logscale y 2
#set yrange [0.0156:128]
set ylabel "Abort Rate" offset 2,0     # Y轴标签
set ytics offset 0.6,0

abr_pth = "../mid/ycsb_diffwd_hc_abr.txt"

set size 0.58,1
set origin 0.445,0.1
set title '(b) Abort Rate'
set xlabel 'Warp Density'
plot abr_pth using 1:2 title "tpl\\\_nw" with linespoints ls 3, \
     abr_pth using 1:3 title "tpl\\\_wd" with linespoints ls 4, \
     abr_pth using 1:4 title "TO" with linespoints ls 5, \
     abr_pth using 1:5 title "MVCC" with linespoints ls 6, \
     abr_pth using 1:6 title "Silo" with linespoints ls 7, \
     abr_pth using 1:7 title "TicToc" with linespoints ls 8

unset multiplot