// assert function for debugging
function assert(condition, message) {
    if (!condition) {
        const error = new Error(`断言失败: ${message}`);
        console.error(
            `%c❌ ${error.message}`,
            'color: red; font-weight: bold; font-size: 14px;'
        );
        throw error;
    }
    console.log(`%c✓ ${message || '检查通过'}`, 'color: green;');
}

// 1. 读取数据
d3.json("example_6.json").then(config => {
    makeChart(config);
}).catch(error => {
    console.error("无法读取JSON数据:", error);
    document.body.innerHTML = "<h3 style='color:red; text-align:center; margin-top:50px'>错误：无法加载JSON数据</h3>";
});

function makeChart(config) {
    // ============================================================
    // 1. 信息提取与配置
    // ============================================================
    const columns = config.data.columns;
    const xColumn = columns.find(c => c.role === 'x');
    const yColumn = columns.find(c => c.role === 'y');
    const groupColumn = columns.find(c => c.role === 'group' || c.role === 'category');
    const imageMap = (config.images && config.images.field) ? config.images.field : {};

    if (!xColumn || !yColumn || !groupColumn) {
        console.error("无法找到必要的列定义");
        return;
    }
    const categoryKey = xColumn.name;
    const valueKey = yColumn.name;
    const groupKey = groupColumn.name;
    const rawData = config.data.data;

    // 数据完整性校验 (Data Integrity Assertion)
    // 1. 获取全局所有的年份/类别 (用于确保整个数据集只有两个对比年份)
    const uniqueGlobalYears = Array.from(new Set(rawData.map(d => d[groupKey])));

    // 校验 A: 全局只能有两个年份 (例如 2019 和 2024)
    assert(uniqueGlobalYears.length === 2,
        `全局数据错误: 检测到 ${uniqueGlobalYears.length} 个年份/类别 (${uniqueGlobalYears.join(", ")}). 此图表仅支持 2 个年份的对比 (Base vs Target).`
    );
    // 2. 按 X轴 (Media Type / Country) 分组检查
    const validationMap = d3.group(rawData, d => d[categoryKey]);
    validationMap.forEach((rows, key) => {
        // 校验 B: 检查是否存在年份重复的数据
        // 我们不只是数个数，先看有没有同一个年份出现了两次
        const yearCounts = d3.rollup(rows, v => v.length, d => d[groupKey]);
        yearCounts.forEach((count, year) => {
            assert(count === 1,
                `数据重复错误: 在 "${key}" 中，年份 "${year}" 出现了 ${count} 次。每个年份只能有一条数据。`
            );
        });

        // 校验 C: 检查是否严格包含 2 条记录
        // (如果通过了校验B，说明没有重复年份，那么这里不等于2就意味着缺年份或者多出了非法年份)
        assert(rows.length === 2,
            `数据数量错误: "${key}" 共有 ${rows.length} 条有效记录. 必须严格为 2 条 (Base vs Target).`
        );

        // 校验 D: 再次确认这两个年份是否就是全局的那两个年份 (防止数据虽然是2条，但年份写错了，比如写成了 2020)
        const itemYears = rows.map(d => d[groupKey]).sort();
        assert(JSON.stringify(itemYears) === JSON.stringify(uniqueGlobalYears),
            `年份匹配错误: "${key}" 的年份是 [${itemYears.join(", ")}], 但全局期望是 [${uniqueGlobalYears.join(", ")}].`
        );
    });


    // 确定年份
    const allYears = Array.from(new Set(rawData.map(d => d[groupKey]))).sort();
    const baseYear = allYears[0];
    const targetYear = allYears[allYears.length - 1];
    // 数据处理
    const groupedDataMap = d3.group(rawData, d => d[categoryKey]);
    let plotData = [];
    groupedDataMap.forEach((values, key) => {
        const baseItem = values.find(d => d[groupKey] === baseYear);
        const targetItem = values.find(d => d[groupKey] === targetYear);
        if (baseItem && targetItem) {
            const v1 = parseFloat(baseItem[valueKey]);
            const v2 = parseFloat(targetItem[valueKey]);
            plotData.push({
                country: key,
                baseYear: baseYear,
                targetYear: targetYear,
                baseVal: v1,
                targetVal: v2,
                diff: v2 - v1,
                maxVal: Math.max(v1, v2),
                flag: imageMap[key]
            });
        }
    });
    plotData.sort((a, b) => b.diff - a.diff);   // 排序：按变化量排序
    // 颜色与样式
    let chartColors = config.colors_dark.available_colors;
    const bgColor = config.colors_dark.background_color || "#000000";
    const fontTitle = config.typography.title.font_family || "Arial";
    const primaryTitleColor = "#ffffff";

    // ============================================================
    // 2. 绘图环境设置 (SVG Setup)
    // ============================================================
    const width = 1200;
    const height = 1600;
    const margin = { top: height*0.1, right: 80, bottom: height*0.1, left: 0 };

    d3.select("body").selectAll("svg").remove();

    const svg = d3.select("body")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .style("background-color", bgColor)
        .style("font-family", fontTitle);

    const chartArea = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${margin.top})`);

    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // ============================================================
    // 3. 比例尺定义与位置计算 (Scales & Positioning)
    // ============================================================
    const maxDataVal = d3.max(plotData, d => d.maxVal);

    // --- A. 定义初始参数 ---
    let maxR = 120;   // 默认最大圆半径
    let minR = 20;   // 默认最小圆半径
    let gap = 15;    // 圆间默认间距
    const rightLabelReserve = 80;   // 预留给右侧Y标签的宽度
    const maxAllowedWidth = innerWidth - rightLabelReserve; // 最大绘图宽度

    // --- B. 试算与自动缩放 (Auto-Scaling) ---
    // 定义一个计算宽度的函数，测试当前尺寸是否会溢出
    function calculateTotalWidth(currentMaxR, currentGap) {
        // 临时建立一个比例尺
        const currentScaleFactor = currentMaxR / 80; // 相对于默认80的缩放比
        const tempScale = d3.scaleSqrt()
            .domain([0, maxDataVal]) // 现在这里能正确读到 maxDataVal 了
            .range([minR * currentScaleFactor, currentMaxR]);
        let cX = 0;
        // 模拟排版
        plotData.forEach((d, i) => {
            const r = tempScale(d.maxVal);
            if (i === 0) {
                cX = 2 * r;
            } else {
                cX = cX + currentGap + r; // 加上间距和新半径(圆心距)
                cX = cX + r; // 加上新半径(到达右边缘)
            }
        });
        return cX;
    }

    // 1. 第一次试算：用默认大小算一下总宽
    let totalContentWidth = calculateTotalWidth(maxR, gap);
    // 2. 检查溢出：如果总宽 > 允许宽度，计算缩放系数
    let scaleFactor = 1;
    if (totalContentWidth > maxAllowedWidth) {
        console.warn(`内容宽度 (${totalContentWidth.toFixed(0)}px) 超出画布，正在自动缩小...`);
        scaleFactor = maxAllowedWidth / totalContentWidth;
    }
    // 3. 应用缩放：修正半径和间距
    maxR = maxR * scaleFactor;
    gap = gap * scaleFactor;

    // --- C. 正式定义比例尺 ---
    const radiusScale = d3.scaleSqrt()
        .domain([0, maxDataVal])
        // 范围也要跟着缩放，保持大小圆的比例
        .range([minR * scaleFactor, maxR]);

    // --- D. 正式计算 X 坐标 ---
    let currentX = 0;
    plotData.forEach((d, i) => {
        const r = radiusScale(d.maxVal);
        d.r = r; // 存下来备用
        if (i === 0) {
            // 第一个元素
            d.xPos = r;
            currentX = 2 * r;
        } else {
            // 后续元素
            d.xPos = currentX + gap + r;
            currentX = d.xPos + r;
        }
    });

    // 更新最终的总宽度 (应该是等于或小于 maxAllowedWidth 的)
    totalContentWidth = currentX;

    // 计算居中偏移量
    const centerOffset = Math.max(0, (innerWidth - totalContentWidth) / 2);


    // --- E. Y轴与颜色 (保持不变) ---
    const maxDiff = d3.max(plotData, d => d.diff) || 0;
    const minDiff = d3.min(plotData, d => d.diff) || 0;

    const absMax = Math.max(Math.abs(maxDiff), Math.abs(minDiff));
    const gridStep = absMax <= 15 ? 2 : 5;

    let yMaxLimit = Math.ceil(maxDiff / gridStep) * gridStep;
    let yMinLimit = Math.floor(minDiff / gridStep) * gridStep;

    if (yMaxLimit > 0) {
        yMaxLimit += gridStep * 0.5;
    }

    let currentSpan = yMaxLimit - yMinLimit;
    if (currentSpan === 0) currentSpan = gridStep * 2;
    let zeroRatio = yMaxLimit / currentSpan;
    // 零点横线位置被限定在 35%-65% 之间
    if (zeroRatio < 0.35) {
        const neededMax = Math.abs(yMinLimit) * (0.35 / 0.65);
        yMaxLimit = Math.ceil(neededMax / gridStep) * gridStep;
    }
    if (zeroRatio > 0.65) {
        const neededMin = -(yMaxLimit * (0.35 / 0.65));
        yMinLimit = Math.floor(neededMin / gridStep) * gridStep;
    }
    if (yMaxLimit === 0 && yMinLimit === 0) {
        yMaxLimit = gridStep;
        yMinLimit = -gridStep;
    }
    if (yMaxLimit < 0) yMaxLimit = 0;
    if (yMinLimit > 0) yMinLimit = 0;
    const yScale = d3.scaleLinear()
        .domain([yMinLimit, yMaxLimit])
        .range([innerHeight, 0]);
    const yZero = yScale(0);
    const colorScale = d3.scaleOrdinal()
        .domain(plotData.map(d => d.country))
        .range(chartColors);


    // ============================================================
    // 4. 绘制网格线 (Grid)
    // ============================================================

    // 生成网格线数据
    // 逻辑：分向上和向下两段，只有真的有柱子达到的区域才画线
    const gridValues = [];
    const visualMax = Math.ceil(maxDiff / gridStep) * gridStep;
    const visualMin = Math.floor(minDiff / gridStep) * gridStep;
    for (let i = 0; i <= yMaxLimit; i += gridStep) {
        if (i === 0 || i <= visualMax) {
            gridValues.push(i);
        }
    }
    for (let i = -gridStep; i >= yMinLimit; i -= gridStep) {
        if (i >= visualMin) {
            gridValues.push(i);
        }
    }

    // 排序确保 0 线在正确位置绘制
    gridValues.sort((a, b) => a - b);

    // 绘制线
    chartArea.append("g")
        .attr("class", "grid")
        .selectAll("line")
        .data(gridValues)
        .enter()
        .append("line")
        .attr("x1", centerOffset - 20)
        .attr("x2", centerOffset + totalContentWidth + 20)
        .attr("y1", d => yScale(d))
        .attr("y2", d => yScale(d))
        .attr("stroke", d => {
            if (d === 0) return "#ffffff"; // 0线：纯白
            return "#333";                 // 其他线：深灰
        })
        .attr("stroke-width", d => {
            if (d === 0) return 3;         // 0线：加粗 (3px)
            return 2;                      // 其他线：标准 (2px)
        })
        .attr("stroke-dasharray", d => {
            if (d === 0) return "none";    // 0线：实线
            return "none";                  // 其他线：虚线"4,4"
        })
        .attr("opacity", d => {
            if (d === 0) return 1;         // 0线：完全不透明
            return 0.5;                    // 其他线：半透明
        });

    // 绘制 Grid 标签
    chartArea.selectAll(".grid-label")
        .data(gridValues)
        .enter()
        .append("text")
        .attr("x", centerOffset + totalContentWidth + 50)
        .attr("y", d => yScale(d))
        .attr("dy", "0.32em")
        .attr("fill", "#666")
        .style("font-family", "'Oswald', sans-serif")
        .style("font-size", "25px")
        .text(d => d === 0 ? "Init" : (d > 0 ? `${d}pp` : `${d}pp`));

    // ============================================================
    // 5. 绘制核心图表 (Drawing) -
    // ============================================================

    // --- A. 定义渐变与计算颜色 (Gradient & Color Definitions) ---
    const defs = svg.append("defs");
    // 【新增】定义一个通用的圆形裁剪路径
    // clipPathUnits="objectBoundingBox" 意味着坐标 0-1 代表宽高 0%-100%
    // cx=0.5, cy=0.5, r=0.5 表示在对象的正中心画一个直径等于对象宽度的圆
    defs.append("clipPath")
        .attr("id", "flag-clip-circle")
        .attr("clipPathUnits", "objectBoundingBox")
        .append("circle")
        .attr("cx", 0.5).attr("cy", 0.5).attr("r", 0.5);
    plotData.forEach(d => {
        const safeId = d.country.replace(/[^a-zA-Z0-9]/g, '');
        const gradId = `grad-${safeId}`;
        // 1. 获取基础颜色 (鲜亮，用于基准圆和描边)
        const baseColor = d3.color(colorScale(d.country));
        // 2. 【核心修改】计算用于填充圆的“实心暗淡色”
        // 我们需要一个看起来像是 30% 透明度，但实际上是不透明的颜色。
        // 通过将黑色背景与鲜亮基准色进行插值计算得到。
        // 这样得到的圆是实心的，不会受底部条带影响出现半深半浅的问题。
        const solidDimColor = d3.interpolateRgb("#000000", baseColor)(0.3);
        d.solidDimColor = solidDimColor; // 存入数据，供圆使用
        // 3. 定义用于渐变的半透明颜色 (保持条带的通透渐变感)
        const gradientDimColor = baseColor.copy();
        gradientDimColor.opacity = 0.3;
        const gradient = defs.append("linearGradient")
            .attr("id", gradId)
            .attr("x1", "0%")
            .attr("y1", "0%")
            .attr("x2", "0%")
            .attr("y2", "100%");
        // 根据涨跌决定渐变方向
        if (d.diff > 0) {
            // 增长: Target在上(暗), Base在下(亮)
            gradient.append("stop").attr("offset", "0%").attr("stop-color", gradientDimColor);
            gradient.append("stop").attr("offset", "100%").attr("stop-color", baseColor);
        } else {
            // 下跌: Base在上(亮), Target在下(暗)
            gradient.append("stop").attr("offset", "0%").attr("stop-color", baseColor);
            gradient.append("stop").attr("offset", "100%").attr("stop-color", gradientDimColor);
        }
    });

    // --- B. 创建分组 (保持不变) ---
    const groups = chartArea.selectAll(".country-group")
        .data(plotData)
        .enter()
        .append("g")
        .attr("class", "country-group")
        .attr("transform", d => `translate(${centerOffset + d.xPos}, 0)`);

    // --- C. 绘制连接柱 (Pill/Bar) - [最终完美版：重叠消除缝隙] ---
    // 逻辑：路径直接从 Base 的圆心位置出发，画到 Target，然后由 Base 圆覆盖接口
    groups.append("path")
        .attr("d", d => {
            const rBase = radiusScale(d.baseVal);   // 基准圆半径
            const rTarget = radiusScale(d.targetVal); // 预测圆半径
            const yBase = yScale(0);    // 基准线 Y (圆心)
            const yTarget = yScale(d.diff); // 预测线 Y (圆心)
            const isGrowth = d.diff > 0;

            // 我们构建一个"圆头梯形"：
            // 一端是平的 (放在 Base 圆心，会被遮住)
            // 一端是圆的 (Target 圆头)

            if (isGrowth) {
                // --- 增长 (Target 在上) ---
                return `
                    M ${-rBase},${yBase}
                    L ${-rTarget},${yTarget}
                    A ${rTarget},${rTarget} 0 0 1 ${rTarget},${yTarget}
                    L ${rBase},${yBase}
                    Z
                `;
                // M: 移动到基准圆圆心左侧
                // L: 连线到预测圆圆心左侧
                // A: 画一个上凸的半圆 (Target的顶)
                // L: 连线回基准圆圆心右侧
                // Z: 闭合 (形成一条水平线横穿基准圆圆心，但这会被 Section D 的圆盖住)
            } else {
                // --- 下跌 (Target 在下) ---
                return `
                    M ${-rBase},${yBase}
                    L ${-rTarget},${yTarget}
                    A ${rTarget},${rTarget} 0 0 0 ${rTarget},${yTarget}
                    L ${rBase},${yBase}
                    Z
                `;
                // A: 画一个下凸的半圆 (Target的底)
            }
        })
        .attr("fill", d => `url(#grad-${d.country.replace(/[^a-zA-Z0-9]/g, '')})`)
        .attr("opacity", 1);

    // --- D. 绘制 2024 圆 (保持不变，鲜亮实心) ---
    groups.append("circle")
        .attr("cy", yZero)
        .attr("r", d => radiusScale(d.baseVal))
        .attr("fill", d => colorScale(d.country))
        .attr("opacity", 1);

    // --- E. 绘制 2029 圆 (仅描边) ---
    // 填充由 Section C 完成，这里只画轮廓
    groups.append("circle")
        .attr("cy", d => yScale(d.diff))
        .attr("r", d => radiusScale(d.targetVal))
        .attr("fill", "none") // 不填充
        .attr("stroke", d => colorScale(d.country))
        .attr("stroke-width", 4)
        .attr("opacity", 0.6); // 稍微淡一点的描边，增加精致感

    // --- F. 文本标签 (Text Labels) - [智能布局版：防重叠 + 外部数值] ---

    // 定义判断"差值微小"的阈值 (比如小于 5pp 就算小)
    const smallDiffThreshold = 5;
    const iconSize = 40;
    const iconPadding = 8;

    // 1. 国家名称与国旗 (组合放置在柱子最顶端)
    const labelGroups = groups.append("g")
        .attr("transform", d => {
            const isGrowth = d.diff > 0;
            // 找出最上面的那个圆 (SVG坐标系中 y 越小越靠上)
            const topY = isGrowth ? yScale(d.diff) : yScale(0);
            const topRadius = isGrowth ? radiusScale(d.targetVal) : radiusScale(d.baseVal);
            let offset = 40;
            if (isGrowth && Math.abs(d.diff) < smallDiffThreshold) {
                offset += 20; // 多留出文字的高度
            }
            return `translate(0, ${topY - topRadius - offset})`;    // 定位到圆顶上方
        });

    // 1.1 绘制灰色背景圆 (始终存在，作为底托)
    labelGroups.append("circle")
        .attr("class", "bg-circle")
        .attr("r", iconSize / 2)
        .attr("fill", "#ccc");


    // 1.2 绘制国旗图标
    labelGroups.filter(d => d.flag)
        .append("image")
        .attr("xlink:href", d => d.flag)
        .attr("width", iconSize).attr("height", iconSize)
        .attr("preserveAspectRatio", "xMidYMid slice").attr("clip-path", "url(#flag-clip-circle)");

    // 1.3 绘制X名称
    labelGroups.append("text")
        .attr("fill", "#fff")
        .style("font-weight", "bold")
        .style("font-size", "25px")
        .style("text-shadow", "0px 0px 4px rgba(0,0,0,0.8)")
        .text(d => d.country)
        .each(function(d) {
            const g = d3.select(this.parentNode);
            const circle = g.select("circle");
            const image = g.select("image");
            const text = d3.select(this);
            // 字体大小自动缩放逻辑
            const maxAllowedWidth = (d.r * 2) + 8;
            let currentTextWidth = this.getComputedTextLength();    //测量当前文字宽度
            // 3. 计算缩放比例
            // 如果文字宽 100px，允许宽 80px，比例就是 0.8
            // 限制最小字号为 12px，防止缩得太小看不见
            let scale = 1;
            if (currentTextWidth > maxAllowedWidth) {
                scale = maxAllowedWidth / currentTextWidth;
            }


            let newFontSize = Math.max(12, 25 * scale);
            text.style("font-size", `${newFontSize}px`);


            const isLongName = d.country.length > 5;
            if (isLongName) {
                // 纵向布局 X，换行，图片
                circle.attr("cx", 0)
                    .attr("cy", -20);
                image.attr("x", -iconSize / 2)
                    .attr("y", -iconSize / 2 - 20);
                text.attr("x", 0)
                    .attr("y", 20)
                    .attr("text-anchor", "middle");
            } else {
                // 横向布局 X，图片并排
                // 1. 计算整体宽度，求起始点
                const textWidth = this.getComputedTextLength();
                const totalWidth = iconSize + iconPadding + textWidth;
                const startX = -totalWidth / 2;
                const cy = -iconSize / 2 + 6;
                circle.attr("cx", startX + iconSize / 2)
                    .attr("cy", cy);
                image.attr("x", startX)
                    .attr("y", cy - iconSize / 2); // 确保图片中心和圆心重合
                text.attr("x", startX + iconSize + iconPadding)
                    .attr("y", 0)
                    .attr("text-anchor", "start");
             }
        });


    // 2. 变化量 (pp) - [修改版：智能外移]
    groups.append("text")
        .attr("y", d => {
            // 判断逻辑：两圆圆心距离-半径和<字体大小，则认为拥挤，移到外部
            const fontSize = 30;
            const yBase = yScale(0);
            const yTarget = yScale(d.diff);
            const centerDistance = Math.abs(yTarget - yBase);
            const rBase = radiusScale(d.baseVal);
            const rTarget = radiusScale(d.targetVal);
            const gap = centerDistance - (rBase + rTarget);
            const isCrowded = gap < fontSize;

            if (!isCrowded) {
                // 情况A: 差值够大，文字放在条带正中间 (原逻辑)
                return (yScale(0) + yScale(d.diff)) / 2;
            } else {
                // 情况B: 差值太小，文字移到外部
                // 方向逻辑：原始圆 -> 预测圆 -> 再往外
                const rTarget = radiusScale(d.targetVal); // 预测圆半径
                const yTarget = yScale(d.diff); // 预测圆圆心
                if (d.diff > 0) {
                    // 增长 (向上): 原始在下，预测在上 -> 往上移 (Top)
                    // 位置 = 预测圆心 - 半径 - 文字缓冲
                    return yTarget - rTarget - 10;
                } else {
                    // 下跌 (向下): 原始在上，预测在下 -> 往下移 (Bottom)
                    // 位置 = 预测圆心 + 半径 + 文字缓冲(考虑到文字基线，要多加点)
                    return yTarget + rTarget + 30;
                }
            }
        })
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .attr("fill", d => {
                // 混合颜色 (基准色 + 白色)
                return d3.interpolateRgb(colorScale(d.country), "#ffffff")(0.4);
        })
        .style("font-size", "35px")
        .style("font-weight", "bold")
        .text(d => {
            if (Math.abs(d.diff) < 0.1) return "0pp"; // 处理近似0
            const pp = d.diff.toFixed(0);
            return pp > 0 ? `+${pp}pp` : `${pp}pp`;
        });

    // 3. 原始圆数值
    groups.append("text")
        .attr("y", yZero)
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .attr("fill", d => {
            // 1. 获取该国家的背景颜色对象
            const color = d3.color(colorScale(d.country));
            const brightness = (color.r * 299 + color.g * 587 + color.b * 114) / 1000;
            // 3. 设定阈值
            // 如果亮度 > 150 (说明是浅色背景，如黄色)，文字用黑色
            return brightness > 200 ? "rgba(0,0,0,0.85)" : "rgba(255,255,255,0.9)";
        })
        .style("font-size", "30px")
        .style("font-weight", "bold")
        .text(d => `${Math.round(d.baseVal)}%`);

    // 4. 预测圆数值
    groups.append("text")
        .attr("y", d => yScale(d.diff))
        .attr("dy", "0.35em")
        .attr("text-anchor", "middle")
        .attr("fill", "rgba(255,255,255,0.95)")
        .style("font-size", "30px")
        .style("font-weight", "bold")
        .text(d => `${Math.round(d.targetVal)}%`);

    // ============================================================
    // 6. 标题与图例 (Auto Layout - 智能区域 & 自动堆叠版)
    // ============================================================

    // --- A. 智能区域判断 (Smart Area Detection) ---

    // 1. 决定上下位置 (基于前 30% 数据的趋势)
    const checkRatio = 0.3; // 检查前 30% 的柱子
    const checkCount = Math.max(2, Math.ceil(plotData.length * checkRatio));
    const leftBars = plotData.slice(0, checkCount);

    // 如果左侧区域中，超过一半的柱子是"涨"的(占据了上方空间)，标题就沉底
    const positiveCount = leftBars.filter(d => d.diff > 0).length;
    const isTitleAtBottom = positiveCount >= (leftBars.length / 2);

    // 2. 设定标题区域的起始 Y 坐标 (Horizon)
    let titleStartY;
    const maxR_l = radiusScale(maxDataVal);
    const zeroLineY = margin.top + yScale(0);
    if (isTitleAtBottom) {
        titleStartY = zeroLineY + maxR_l + 80;
        titleStartY = Math.min(titleStartY, height - margin.bottom - 150);
        titleStartY = Math.max(titleStartY, height * 0.55);
    } else {
        titleStartY = margin.top + 20;
    }

    // 3. 计算可用宽度 (Collision Detection - 基于涨跌性质)
    // 逻辑变更：
    // - 如果标题沉底(Bottom)，说明左侧主要是涨的，宽度延伸直到遇到第一个"跌"的柱子。
    // - 如果标题置顶(Top)，说明左侧主要是跌的，宽度延伸直到遇到第一个"涨"的柱子。
    let collisionX = width - margin.right;
    for (let i = 0; i < plotData.length; i++) {
        const d = plotData[i];
        let stopHere = false;
        if (isTitleAtBottom) {
            // 【沉底模式】：标题属于"上涨区"
            // 只要遇到第一个"跌"的 (diff < 0)，就是边界
            if (d.diff < 0) {
                stopHere = true;
            }
        } else {
            // 【置顶模式】：标题属于"下跌区"
            // 只要遇到第一个"涨"的 (diff > 0)，就是边界
            if (d.diff > 0) {
                stopHere = true;
            }
        }
        if (stopHere) {
            // 找到了边界柱子,边界 = 柱子中心 - 柱子半径 - 额外留白(40px)
            const r = radiusScale(d.maxVal);
            collisionX = centerOffset + d.xPos - r - 40;
            break; // 停止循环，不再往后找了
        }
    }
    collisionX = Math.min(collisionX, width * 0.6);

    // 计算最终可用宽度
    let availableWidth = Math.max(300, collisionX - margin.left);
    console.log("标题区域可用宽度:", availableWidth);
    const scaleFactor_l = Math.min(1, availableWidth / 500);


    // --- B. 绘制标题内容 (自动堆叠) ---

    let currentY = titleStartY;

    // 1. 主标题
    const titleSize = 80 * scaleFactor_l; // 动态字号
    const titleText = svg.append("text")
        .attr("x", margin.left + 100)
        .attr("y", currentY) // 注意：SVG text y 是基线，wrap函数会调整它
        .style("font-size", `${Math.max(40, titleSize)}px`) // 最小40px
        .style("font-weight", "bold")
        .attr("fill", primaryTitleColor)
        .text(config.titles.main_title || "G7 Debt Projections");

    // 自动换行
    wrap(titleText, availableWidth, 1.1);

    // 获取换行后的实际高度，更新 currentY
    let bbox = titleText.node().getBBox();
    currentY += bbox.height + 20; // 加上高度和间距

    // 2. 副标题
    const subSize = 30 * scaleFactor_l;
    const subText = svg.append("text")
        .attr("x", margin.left + 100)
        .attr("y", currentY)
        .style("font-size", `${Math.max(20, subSize)}px`)
        .attr("fill", "#888")
        .text((config.titles.sub_title || ""));

    wrap(subText, availableWidth*0.95, 1.2);

    bbox = subText.node().getBBox();
    currentY += bbox.height + 30; // 间距大一点给 Insights

    // 3. Main Insights
    const mainInsightsText = config.titles.main_insights;
    if (mainInsightsText) {
        const insightSize = 20 * scaleFactor_l;
        const insightsTextElem = svg.append("text")
            .attr("x", margin.left + 100)
            .attr("y", currentY)
            .style("font-size", `${Math.max(16, insightSize)}px`)
            .style("font-weight", "bold")
            .attr("fill", "#fff")
            .text(mainInsightsText);

        wrap(insightsTextElem, availableWidth*0.95, 1.1);
    }

    // --- C. 绘制图例 (Legend) ---

    // 1. 位置计算
    const legendCheckCount = Math.min(plotData.length, 2);
    const rightBars = plotData.slice(-legendCheckCount);
    const rightPositiveCount = rightBars.filter(d => d.diff > 0).length;
    const isLegendAtTop = rightPositiveCount <= (legendCheckCount / 2);

    // 图例总体位置
    // 计算X位置
    const legendBoxWidth = 260;
    const rulerX = centerOffset + totalContentWidth + 50;
    const legendX = rulerX - legendBoxWidth - 30;
    let legendY;
    if (isLegendAtTop) {
        legendY = margin.top;
    } else {
        legendY = height - margin.bottom - 280;
    }

    const legendG = svg.append("g")
        .attr("transform", `translate(${legendX}, ${legendY})`);

    // 2. 定义图例渐变
    const legendGradId = "legend-gradient";
    if (svg.select(`#${legendGradId}`).empty()) {
        const legendGradient = defs.append("linearGradient")
            .attr("id", legendGradId)
            .attr("x1", "0%").attr("y1", "0%")
            .attr("x2", "0%").attr("y2", "100%");
        legendGradient.append("stop").attr("offset", "0%").attr("stop-color", "#ffffff").attr("stop-opacity", 1);
        legendGradient.append("stop").attr("offset", "100%").attr("stop-color", "#ffffff").attr("stop-opacity", 0.3);
    }

    // 3. 绘制背景框
    legendG.append("rect")
        .attr("width", 260)
        .attr("height", 320)
        .attr("fill", "none")
        .attr("stroke", "#fff")
        .attr("rx", 10);

    // 4. 标题 "HOW TO READ"
    legendG.append("text")
        .attr("x", 30)
        .attr("y", 50)
        .attr("fill", "#fff")
        .style("font-family", "'Oswald', sans-serif")
        .style("font-weight", "700")
        .style("font-size", "24px")
        .text("HOW TO READ");

    // 5. 绘制图形
    const legX = 80;
    const legY1 = 110;
    const legY2 = 200;
    const r1 = 28;
    const r2 = 20;

    // 5.1 路径
    const pillPath = `
        M ${legX - r1}, ${legY1}
        L ${legX - r2}, ${legY2}
        A ${r2},${r2} 0 0 0 ${legX + r2},${legY2}
        L ${legX + r1}, ${legY1}
        Z
    `;
    legendG.append("path")
        .attr("d", pillPath)
        .attr("fill", `url(#${legendGradId})`)
        .attr("opacity", 0.8);

    // 5.2 圆形
    legendG.append("circle").attr("cx", legX).attr("cy", legY1).attr("r", r1).attr("fill", "#fff");
    legendG.append("circle")
        .attr("cx", legX).attr("cy", legY2).attr("r", r2)
        .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 4).attr("opacity", 0.6);

    // 6. 绘制说明文字 (自动换行)

    const textBaseX = 130;
    const maxTextWidth = 110;   // 最大允许宽度

    const labelStyle = (selection) => {
        selection.attr("x", textBaseX).attr("fill", "#ddd").style("font-family", "'Oswald', sans-serif").style("font-size", "22px");
    };
    const unitStyle = (selection) => {
        selection.attr("x", textBaseX).attr("fill", "#888").style("font-family", "'Oswald', sans-serif").style("font-size", "20px").style("font-weight", "400");
    };

    // 6.1 Base Year
    legendG.append("text")
        .call(labelStyle)
        .attr("y", legY1)
        .attr("dy", "-0.2em")
        .style("font-weight", "700")
        .text(baseYear);

    // 6.2 Base Unit (应用自动换行)
    legendG.append("text")
        .call(unitStyle)
        .attr("y", legY1)
        .attr("dy", "1em")
        .text(`(${valueKey})`)
        .call(wrap, maxTextWidth, 1.1);

    // 6.3 Target Year
    legendG.append("text")
        .call(labelStyle)
        .attr("y", legY2)
        .attr("dy", "-0.2em")
        .style("font-weight", "700")
        .text(`${targetYear}`);

    // 6.4 Target Unit (应用自动换行)
    legendG.append("text")
        .call(unitStyle)
        .attr("y", legY2)
        .attr("dy", "1em")
        .text(`(${valueKey})`)
        .call(wrap, maxTextWidth, 1.1);

}

// 【辅助函数】SVG文本自动换行
// text: D3文本选择集
// width: 允许的最大宽度 (像素)
// lineHeightEm: 行高 (倍数, 例如 1.2)
function wrap(text, width, lineHeightEm) {
    text.each(function () {
        var text = d3.select(this),
            words = text.text().split(/\s+/).reverse(), // 将文本按空格拆分为单词数组
            word,
            line = [],
            lineNumber = 0,
            lineHeight = lineHeightEm || 1.1, // 默认行高
            y = text.attr("y"),
            x = text.attr("x"),
            // 获取初始的 dy 值，如果没有则默认为 0
            dy = parseFloat(text.attr("dy") || 0),
            // 创建第一个 tspan
            tspan = text.text(null).append("tspan").attr("x", x).attr("y", y).attr("dy", dy + "em");

        // 逐个单词尝试添加到当前行
        while (word = words.pop()) {
            line.push(word);
            tspan.text(line.join(" "));
            // 如果当前行宽度超过了最大宽度
            if (tspan.node().getComputedTextLength() > width) {
                line.pop(); // 移除刚才添加的单词
                tspan.text(line.join(" ")); // 恢复当前行文本
                line = [word]; // 将该单词作为下一行的开始
                // 创建新的 tspan，并增加 dy 值以实现换行
                tspan = text.append("tspan")
                    .attr("x", x)
                    .attr("y", y)
                    .attr("dy", ++lineNumber * lineHeight + dy + "em")
                    .text(word);
            }
        }
    });
}
