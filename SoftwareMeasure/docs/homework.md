软件度量—期末大作业
（图片部分缺少）
基本介绍
本次大作业需要同学们对真实的程序项目进行评测和度量。同学们在完成大作业的过程中应该注意以下几个时间点：
口头报告：14周,15周,16周；
实验报告提交：第16周；
小组组队原则上以4人为一组；
具体要求
	我们精心挑选了20个项目(见附录)作为评测的对象，同学们需要从中挑选不少于3个的项目进行漏洞发现和挖掘。
	每个项目采取至少一种动态测试技术和至少一种静态分析技术来进行漏洞挖掘。我们在附录里详细介绍了两款动态测试工具的使用方式以及其他流行的fuzzing工具，同学们可以用来参考。
	（开放题）在完成上述要求的基础上，鼓励同学们探索智能体（Agent）技术在漏洞挖掘中的应用，尝试将其用于测试用例生成、漏洞挖掘、PoC验证复现等环节。
口头报告
报告时间10分钟，届时需准备PPT，并有简单的提问。
实验报告
实验报告要求将对所选择项目的评测过程进行完整的阐述。描述清楚自己所评测的对象，评测所使用的方法，工具和评测的环境，评测的流程和评测的结果。如果发现了项目中的问题或安全漏洞，要求能确定问题的具体位置，类型，产生的原因以及可能造成的危害。如果没有发现问题，请解释没有发现问题的原因以及如何对评测方法进行改进。

实验报告要求：
1. 报告不超过10页。
2. 无需封面、目录和大段文字。
3. 报告内容需包括 (1). 每个项目的手写的driver/fuzzer的截图; (2). 每个项目跑12小时模糊测试的截图（工具运行截图和工具的覆盖趋势图均可）; (3). 每个项目的静态分析工具运行截图。
4. （加分项）如在最新版的项目中找到bug，经调试确认为项目本身的bug，请将bug提交给维护者，并在报告中附上提交issue的链接。
5. （开放题）介绍Agent工具流的设计，例如 Agent 在漏洞挖掘流程中的具体作用、使用的工具或模型、任务分解方式、与动态测试/静态分析工具的结合方式、Agent设计思路等。

评分将主要考虑以下因素：
	是否发现了真实有效的漏洞；
	所选择评测项目的规模和复杂性；
	所编写的测试工具的有效性和工作量；
	口头报告的质量；
	实验报告的质量；
 
附录
i.	待评测的项目（star越多越值得测）
项目名	描述
SerenityOS	小型操作系统
https://github.com/SerenityOS/serenity

zlib	数据压缩软件
https://github.com/madler/zlib

libvpx	视频编码解码压缩库
https://github.com/webmproject/libvpx 

libpcap	网络流量处理库
https://github.com/the-tcpdump-group/libpcap 

curl	 url访问工具
https://github.com/curl/curl 

libmicrodns	小型的mDNS解析器库
https://github.com/videolabs/libmicrodns

cJSON	Json 解析器
https://github.com/DaveGamble/cJSON 

libva	Intel的视频加速库
https://github.com/intel/libva

libexpat	XML解析器
https://github.com/libexpat/libexpat

libass	字幕格式的便携式字幕渲染器
https://github.com/libass/libass

libxlsxwriter	 Excel XLSX 文件处理库
https://github.com/jmcnamara/libxlsxwriter

libplist	用二进制或XML处理Apple Property List格式的库
https://github.com/libimobiledevice/libplist

libgd	GD 图像处理库
https://github.com/libgd/libgd

libsndfile	用于读取和写入包含采样音频数据的声音文件的C库
https://github.com/erikd/libsndfile

libucl	通用配置库解析器
https://github.com/vstakhov/libucl

libpng	便携式网络图形支持库
https://github.com/glennrp/libpng

libconfig	Configurations 文件处理库
https://github.com/hyperrealm/libconfig

SmallerC	小型的C编译器
https://github.com/alexfru/SmallerC

Xiangshan	高性能RISC-V CPU (值得测)
https://github.com/OpenXiangShan/XiangShan 

CVA6	开源RISC-V CPU
https://github.com/openhwgroup/cva6 


 
ii.	AFL (American Fuzzy lop) 使用介绍
简单介绍
AFL号称是当前最高级的Fuzzing测试工具之一，由lcamtuf所开发。可在http://lcamtuf.coredump.cx/afl/进行简单了解和下载。AFL通过对源码进行重新编译时进行插桩（简称编译时插桩）的方式自动产生测试用例来探索二进制程序内部新的执行路径。
使用方法(以项目libjson为例)
	安装
在适当位置，直接运行如下脚本
                              










	下载项目文件
 
	编译libjson
指定编译使用AFL的编译器
此过程可以通过引入环境变量或者修改编译脚本来实现。
针对libjson项目，原始的编译脚本如下图

 
原始编译脚本(CC=gcc)
修改后
 
修改后编译脚本(CC=$(AFL_HOME)/afl-gcc)

      编译
    make clean
    make
	选择被fuzz的程序





	选择输入文件
此项目可以在tests文件夹下找到用于测试的输入文件





将这些文件全部复制到一个输入文件夹中
mkdir IN_ALL
cp tests/bad/* IN_ALL/
cp tests/good/* IN_ALL/
AFL可以在fuzzing前缩小输入的数量，需使用afl-cmin工具，如下图
      











目前的输入文件夹为IN
	开始fuzzing
afl-fuzz -i IN -o OUT ./jsonlint --verify @@
(通过afl-fuzz启动应用程序，-i指定应用程序的输入，-o指定fuzzing结果的输出，jsonlint是可执行程序名，--verify是jsonlint的某个运行参数, @@ @@表示从文件中读入) 。


启动afl-fuzz中往往会报错，表示某些环境变量没有配置或者配置错误，只需根据提示修改即可。如
 
没有打开错误转储机制，执行命令：
sudo su
echo core >/proc/sys/kernel/core_pattern

重新执行，AFL的运行界面如下
 

界面右上角uniq crashes说明了AFL发现的独特的崩溃数。界面上的其他指标本文档不做详细介绍，具体细节可参看AFL的readme文件。
	结果处理
1.	查看crash文件
libjson的运行结果存储在OUT文件夹下，其中的crashes文件夹存储了使程序产生独特崩溃的输入。由于libjson暂未发现问题，故之后换用项目openssl进行说明。
打开crash目录，可以看到crash文件
 

将crash文件用作输入可以使程序崩溃 ，然后gdb调试即可
2.	缩减crash input的大小
使用afl-tmin工具可以缩减crash input的大小，缩减之后更利于我们分析crash的成因，使用方式如下图


















3.	运行崩溃探索模式
此模式会利用已有的崩溃文件作为输入，变异它们以发现新的崩溃。需要使用-C选项，如
./afl-fuzz -i OUT/crashes/ -o peruvian_crashes -m 30000000 -C ./openssl-1.0.1f-afl
	性能分析
输出文件夹下的plot_data保留了fuzz过程中的一些指标随时间的变化信息。运行afl-plot程序可以做出一些图像如下
 
 
	其他
1.	可以使用ASAN


方法
export AFL_USE_ASAN=1
添加 CFLAGS=”-g -fsanitize=address” 到 makefile文件中
使用ASAN编译选项尽量编译成32位系统程序(-m32), 因为Address Sanitize使用Shadow Memory机制，在32机器上需要大约800M的内存，但是在x86_64系统上需要大约20TB的内存。运行时添加内存限制，-m 30000000
2.	可以使用用户字典
如-x $AFL_HOME/dictionaries/json.dict
iii.	libFuzzer使用介绍
简单介绍
LibFuzzer是一个持续执行，覆盖率导向的，演进式的模糊测试引擎，是LLVM项目的一部分。LibFuzzer与被测试的库链接，并通过特定的模糊入口（也称为“目标函数”）将模糊输入的样例集提供给library。然后模糊器跟踪代码会执行到哪些模板，并且在输入数据的语料库上产生突变，以便最大化代码覆盖度。libFuzzer的代码覆盖信息由LLVM的SanitizerCoverage指令执行提供。
libFuzzer 与AFL 的区别在于：libFuzzer 倾向于对某个功能或者函数来进行Fuzzing，AFL 则是对整体程序进行Fuzzing。	LLVMFuzzerTestOneInput() 函数是libFuzzer 库进行单元测试的调用点(模糊入口，目标函数),这里面的实现就是我们将要编写的测试驱动。libFuzzer 能够更深入去挖掘潜在的漏洞,但是需要对源码有一定的了解。
使用方法
	安装和使用
安装方法和简单的教程请参见： https://github.com/google/fuzzer-test-suite/blob/master/tutorial/libFuzzerTutorial.md
注：就libjson项目而言，在编译库时，修改其makefile文件如下
 
执行make clean && make
之后需要编译测试驱动target.c
clang -g -fsanitize=address -fsanitize-coverage=trace-pc-guard target.c –c
之后将库，驱动和libFuzzer的目标文件链接在一起即可形成fuzzer，执行
clang++ -g -fsanitize=address -fsanitize-coverage=trace-pc-guard json.o target.o libFuzzer.a -o app
运行时的效果如下图
 
	测试驱动的编写
使用libFuzzer最重要的部分在于测试驱动的编写，即编写LLVMFuzzerTestOneInput()函数。本文档仅对libjson为例说明如何编写一个测试驱动。
编写驱动需要对程序源码有一定的了解，如果想快速编写一个测试驱动，最简单的办法是从原项目的示例代码或者测试代码转换而来。
就libjson项目而言，我们可以把jsonlint.c转换为我们的驱动。
通过阅读文档，libjson的主要API有









jsonlint.c中的main给出了初始化的方法

 
do_format函数给出了一些结束时可以使用的办法
 
结合它们，我们可以写出一个简单的驱动如下
 
使用AFL来对libFuzzer编写的测试用例(驱动)进行测试
    使用AFL也可以对libFuzzer的测试用例(驱动)进行测试，此时需要新增加两个文件的使用，包括afl-llvm-rt.o.c(位于AFL主文件夹下的llvm_mode文件夹内) 以及afl_driver.cpp(http://llvm.org/doxygen/afl__driver_8cpp_source.html).
仍然以libjson为例，只需要对编译过程进行微小调整。
    第一步，用-fsanitize-coverage=trace-pc-guard选项编译target.c文件，即
clang -g -fsanitize=address -fsanitize-coverage=trace-pc-guard target.c –c
第二步，编译afl-llvm-rt.o.c
clang -c -w $AFL_HOME/llvm_mode/afl-llvm-rt.o.c
第三步，链接target.o  json.o afl-llvm-rt.o.o 和afl_driver.cpp
clang++ -g -fsanitize=leak,address afl_driver.cpp target.o  json.o afl-llvm-rt.o.o -o app
iv.	其他fuzzer
AFLFast
https://github.com/mboehme/aflfast

AFLGo
https://github.com/aflgo/aflgo

SAFL
https://github.com/hghwng/tools-date
SAFL演示视频https://youtu.be/LkiFLNMBhVE

honggfuzz
https://github.com/google/honggfuzz


v.	静态分析工具推荐
Clang Static Analyzer
https://clang-analyzer.llvm.org/ 
Cppcheck
https://clang-analyzer.llvm.org/ 

