// chart.js

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
// choose from 7201 11977 aug_cnc_4671 aug_cnc_4968 aug_cnc_5424
d3.json("7201.json").then(config => {
    makeChart(config);
}).catch(error => {
    console.error("无法读取JSON数据:", error);
    document.body.innerHTML = "<h3 style='color:red; text-align:center; margin-top:50px'>错误：无法加载JSON数据</h3>";
});

function makeChart(config) {
    // ============================================================
    // 1. 信息提取阶段 (Explicit Data Extraction)
    // ============================================================
    const theme = "dark"                // light/dark，仅支持dark
    const use_default_color = true;     // 是否使用自搭配色
    // --- A. 提取字段 ---
    const columns = config.data.columns;
    const xColumn = columns.find(c => c.role === 'x');
    const yColumn = columns.find(c => c.role === 'y');
    const groupColumn = columns.find(c => c.role === 'group');
    assert(xColumn, "找到'x'字段");
    assert(yColumn, "找到'y'字段");
    assert(groupColumn, "找到'group'字段");
    const categoryKey = xColumn.name;   // 分类字段 (Category)
    const valueKey = yColumn.name;      // 数值字段 (Value)
    const groupKey = groupColumn.name;  // 分组字段 (Group)
    // --- B. 提取数据内容 ---
    const rawData = config.data.data;
    const insights = config.metadata.main_insight || [];
    const main_title_content = config.titles.main_title;
    const sub_title_content = config.titles.sub_title;
    const imageMap = (config.images && config.images.field) ? config.images.field : {};
    // --- C. 提取样式信息 ---
    const colors = theme == "light" ? config.colors : config.colors_dark;
    const typeTitle = config.typography.title;
    const typeDesc = config.typography.description;
    const typeLabel = config.typography.label;
    const typeAnnot = config.typography.annotation;
    // 颜色信息
    const myCustomPalette = [
        "#f44336", // Red
        "#9c27b0", // Purple
        "#3f51b5", // Indigo
        "#03a9f4", // Light Blue
        "#009688", // Teal
        "#8bc34a", // Light Green
        "#ffeb3b", // Yellow
        "#ff9800"  // Orange
    ];
    background_color = colors.background_color
    text_color = colors.text_color
    chart_colors = colors.available_colors
    if (use_default_color){
        background_color = "#17325b";
        text_color = "#ffffff";
        chart_colors = myCustomPalette;
    }


    // ============================================================
    // 2. 环境准备 (Setup)
    // ============================================================
    // 设置尺寸和边距
    const width = 1200;
    const height = 1600;
    const margin = { top: 150, right: 50, bottom: 400, left: 50 };
    // --- 定义布局边界 ---
    const sidebarWidth = 100; // 左侧留给文字和图例的宽度
    const headerHeight = 100; // 顶部留给标题的高度
    const padding = 40;       // 元素之间的安全间距

    // --- 计算圆心位置 ---
    const centerX = width * 0.55; // 右侧区域的水平中心
    const centerY = height * 0.6; // 下方区域的垂直中心
    // 严格计算最大半径 (防止碰撞),计算圆心到四个“死线”的距离
    // const distToSidebar = centerX - sidebarWidth; // 到左侧文字区的距离
    const distToRightEdge = width - centerX + 100;      // 到右边缘的距离
    const distToHeader = centerY - headerHeight + 500;  // 到顶部标题区的距离
    const distToBottomEdge = height - centerY;    // 到底边缘的距离
    const innerRadius = 0;
    const maxRadius = Math.min(distToRightEdge, distToHeader, distToBottomEdge) - padding;  // 圆最大半径
    // 设置网页整体背景
    d3.select("body")
        .style("background-color", "#ccc")
        .style("margin", "0")
        .style("display", "flex")
        .style("justify-content", "center")
        .style("padding", "20px");
    // 清空并创建 SVG
    const container = d3.select("#poster-container");
    container.html("");
    const svg = container.append("svg")
        .attr("width", width)
        .attr("height", height)
        .attr("viewBox", `0 0 ${width} ${height}`)
        .style("background-color", background_color)
        .style("box-shadow", "0 0 30px rgba(0,0,0,0.3)");
    // 获取图片，若无图片则使用灰色替代
    const defs = svg.append("defs");    // 映射表，用于后续直接获取 fill 属性
    const iconFillMap = {};  // 提取所有唯一的 Group 名称
    const allGroupsUnique = [...new Set(rawData.map(d => d[groupKey]))];
    const randomSuffix = Math.floor(Math.random() * 10000);
    const sanitizeId = (str) => `icon-pat-${str.replace(/[^a-zA-Z0-9-_]/g, '-')}-${randomSuffix}`;
    allGroupsUnique.forEach(gName => {
        const imgUrl = imageMap[gName];
        if (imgUrl && imgUrl.trim() !== "") {
            // --- 情况 A: 有图片，定义 Pattern ---
            const patternId = sanitizeId(gName);
            // 在 defs 中追加一个 pattern
            // patternContentUnits="objectBoundingBox" 让内部坐标变成 0~1 的相对值
            // 这样无论后面的圆多大，图片都会自动充满
            const pattern = defs.append("pattern")
                .attr("id", patternId)
                .attr("patternUnits", "objectBoundingBox")
                .attr("width", "1")
                .attr("height", "1")
                .attr("patternContentUnits", "objectBoundingBox");
            // 追加白色背景 (防止透明PNG显示出底下的线条)
            pattern.append("rect")
                .attr("width", 1)
                .attr("height", 1)
                .attr("fill", "#fff");
            // 追加图片
            pattern.append("image")
                .attr("href", imgUrl)
                .attr("x", 0)
                .attr("y", 0)
                .attr("width", 1)
                .attr("height", 1)
                // 关键：保持比例并切除多余部分，填满容器
                .attr("preserveAspectRatio", "xMidYMid slice");
            // 记录映射：填充值为 url(...)
            iconFillMap[gName] = `url(#${patternId})`;
        } else {
            // --- 情况 B: 无图片，记录纯灰颜色 ---
            // 记录映射：填充值为 颜色字符串
            iconFillMap[gName] = "#bbb";
        }
    });

    // ============================================================
    // 3. 数据处理 (Data Processing)
    // ============================================================

    // 数据排序逻辑
    // 1. 先按 Group 聚类 (按每个group的最大值降序)
    // 2. 组内按 Value (y) 降序
    // 3. (如果有相同y) 尽可能将相同 x 错开 -> 这一步在排序中比较难“穿插”，
    //    通常我们保证 x 不相邻即可。如果数据量不大，简单排序即可满足。
    //    这里的逻辑实现为：Group -> Value(desc) -> Category(asc)
    // 计算每个group的最大值
    const groupMaxValues = d3.rollup(rawData,
        v => d3.max(v, d => d[valueKey]),
        d => d[groupKey]
    );
    const sortedData = [...rawData].sort((a, b) => {
        // 扇形展示的排序函数
        const groupA = a[groupKey];
        const groupB = b[groupKey];
        // 获取两个数据所属 Group 的最大值
        const maxValGroupA = groupMaxValues.get(groupA) || 0;
        const maxValGroupB = groupMaxValues.get(groupB) || 0;
        // 【规则1】首先按 Group 的最大值降序 (Group 整体排序)
        if (maxValGroupA !== maxValGroupB) {
            return maxValGroupB - maxValGroupA;
        }
        // 如果 Group 最大值碰巧一样，按 Group 名字排，保证同组数据不被打散
        if (groupA < groupB) return -1;
        if (groupA > groupB) return 1;
        // 【规则2】Group 内部，按 Value 降序
        if (b[valueKey] !== a[valueKey]) {
            return b[valueKey] - a[valueKey];
        }
        // 【规则3】Value 相同，按 Category 排序 (起到错开 x 的作用)
        if (a[categoryKey] < b[categoryKey]) return -1;
        if (a[categoryKey] > b[categoryKey]) return 1;
        return 0;
    });

    // ============================================================
    // 4. 比例尺与旋转计算 (Scales & Rotation)
    // ============================================================

    // 角度比例尺 (标准，未旋转)
    const angleScale = d3.scaleBand()
        .domain(sortedData.map((d, i) => i))
        .range([0, 2 * Math.PI])
        .padding(0);
    // 半径比例尺
    const xMax = d3.max(rawData, d => d[valueKey]);
    const radiusScale = d3.scaleSqrt()  // d3.scaleLinear() 为线性比例尺
        .domain([0, xMax])
        .range([innerRadius, maxRadius]);
    // 颜色比例尺
    const allCategories = [...new Set(rawData.map(d => d[categoryKey]))];
    const colorScale = d3.scaleOrdinal()
        .domain(allCategories)
        .range(chart_colors);

    // 样式：定义背景的渐变滤镜
    const deepSpaceColor = "#051025";
    const bgGlowGradient = defs.append("radialGradient")
        .attr("id", "center-light-glow")
        .attr("cx", "50%") // 渐变中心在圆心
        .attr("cy", "50%")
        .attr("r", "50%")  // 渐变半径覆盖整个圆
        .attr("fx", "50%") // 焦点也在圆心
        .attr("fy", "50%");
    // Stop 0%: 圆心位置 (最亮)
    // 使用淡淡的青白色，透明度低一点，叠加在深蓝背景上就是提亮效果
    bgGlowGradient.append("stop")
        .attr("offset", "0%")
        .attr("stop-color", "rgba(200, 240, 255, 0.25)") // 浅青色，半透明
        .attr("stop-opacity", 1);
    // Stop 60%: 中间过渡
    bgGlowGradient.append("stop")
        .attr("offset", "60%")
        .attr("stop-color", "rgba(100, 180, 255, 0.05)") // 颜色变深变淡
        .attr("stop-opacity", 1);
    // Stop 100%: 边缘位置 (完全透明)
    // 确保边缘完全融入背景，看不出边界
    bgGlowGradient.append("stop")
        .attr("offset", "100%")
        .attr("stop-color", deepSpaceColor) // 可以设置成背景色
        .attr("stop-opacity", 0);           // 关键是透明度为 0


    // 定义高斯模糊滤镜，用于制作“泛边框”的光晕效果
    const filter = defs.append("filter")
        .attr("id", "glow-blur")
        .attr("x", "-50%")
        .attr("y", "-50%")
        .attr("width", "200%")
        .attr("height", "200%");
    filter.append("feGaussianBlur")
        .attr("stdDeviation", "10") // 模糊程度，数值越大光晕越散
        .attr("result", "coloredBlur");
    // 定义渐变
    sortedData.forEach((d, i) => {
        const mainColor = colorScale(d[categoryKey]);   // 获取当前扇形主色调
        const gradientId = `sector-gradient-${i}`;      // 生成唯一ID
        const r = radiusScale(d[valueKey]);
        // 创建径向渐变
        const radialGradient = defs.append("radialGradient")
            .attr("id", gradientId)
            .attr("gradientUnits", "userSpaceOnUse")
            .attr("cx", 0).attr("cy", 0).attr("r", r)   // 渐变中心，焦点，半径
        // Stop 1: 圆心附近 (0%) - 设置为背景色，或者深色透明
        // 为了达到最佳融合效果，这里使用背景色，并给予一定的透明度
        radialGradient.append("stop")
            .attr("offset", "0%") // 从圆心开始
            .attr("stop-color", background_color) // 背景色
            .attr("stop-opacity", 1);
        // Stop 2: 保持黑暗区域 (例如到 35%)
        // 这意味着半径前 35% 的区域完全是背景色，制造一个明显的“黑洞”中心
        radialGradient.append("stop")
            .attr("offset", "50%") // 调整这个数字：越大，中心黑洞越大
            .attr("stop-color", background_color)
            .attr("stop-opacity", 1);
        // Stop 3: 中间过渡点 (例如 60% 或 70% 的位置)
        // 这个点让靠近中心的大部分区域保持较暗的状态
        radialGradient.append("stop")
            .attr("offset", "75%")
            .attr("stop-color", d3.color(mainColor).darker(1.5).formatHex()) // 比主色调深很多的颜色，或者继续用背景色过渡
            .attr("stop-opacity", 0.95);
        // Stop 4: 外缘 (100%) - 设置为鲜艳的主色调
        radialGradient.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", mainColor) // 鲜艳的主色
            .attr("stop-opacity", 1);
    });

    // 旋转计算 (Rotation)
    // 计算旋转偏移量，找到最大值的项及其索引
    let maxVal = -Infinity;
    let maxIdx = -1;
    sortedData.forEach((d, i) => {
        if (d[valueKey] > maxVal) {
            maxVal = d[valueKey];
            maxIdx = i;
        }
    });
    // 计算该扇形目前的中心角度 (Current Angle)
    const bandwidth = angleScale.bandwidth();
    const currentAngle = angleScale(maxIdx) + bandwidth / 2;
    // 计算目标角度 (Target Angle)
    const dx = 0 - centerX; // 指向左
    const dy = 0 - centerY; // 指向上
    const targetAngle = Math.atan2(dx, -dy);
    const rotationOffset = targetAngle - currentAngle;

    // ============================================================
    // 7. 绘图 (Drawing)
    // ============================================================

    // --- A. 标题，副标题与insights ---
    // 1. 定义布局参数
    const titleAreaX = width * 0.4; // 从画布中间开始
    const titleAreaY = 80;          // 顶部起始 Y
    const titleAreaW = (width * 0.6) - margin.right; // 宽度 = 右半边 - 边距
    const titleAreaH = 200;         // 标题区域最大高度 (超过这个高度字体就会变小)
    // 创建容器，定位到右半区起点
    const titleG = svg.append("g")
        .attr("transform", `translate(${titleAreaX}, ${titleAreaY})`);

    // --- 2. 绘制主标题 (Main Title) ---
    // 预估分配给主标题的高度 (例如占总高度的 60%)
    const mainTitleH = titleAreaH * 0.6;
    const centerX_title = titleAreaW / 2;

    const mainTitleText = titleG.append("text")
        .style("font-family", typeTitle.font_family)
        .style("font-weight", typeTitle.font_weight)
        .style("fill", text_color)
        .attr("text-anchor", "middle") // 左对齐 (相对于 width/2)
        .attr("dominant-baseline", "hanging"); // 从顶部开始画

    // 调用核心自适应函数
    const actualMainHeight = fitTextByWordCount(
        mainTitleText,
        main_title_content,
        4,  // 单行最多的单词数
        titleAreaW,
        mainTitleH,
        80, // 最大字号
        centerX_title
    );

    // --- 3. 绘制副标题 (Sub Title) ---
    // 副标题放在主标题下面，有一点间距
    const subTitleY = actualMainHeight + 20;
    const subTitleH = titleAreaH - subTitleY; // 剩下的高度给副标题

    const subTitleText = titleG.append("text")
        .attr("transform", `translate(0, ${subTitleY})`)
        .style("font-family", typeDesc.font_family)
        .style("font-weight", typeDesc.font_weight)
        .style("fill", text_color)
        .style("opacity", 0.7)
        .attr("text-anchor", "start")
        .attr("dominant-baseline", "hanging");

    // 调用核心自适应函数
    fitTextByWordCount(
        subTitleText,
        sub_title_content,
        10, // 单行允许最多的字数
        titleAreaW,
        subTitleH,
        40, // 最大字号
        0
    );


    // --- B. 图表主体 ---
    const chartG = svg.append("g")
        .attr("transform", `translate(${centerX}, ${centerY})`);

    // 0. 绘制最底层的背景光晕
    chartG.append("circle")
        // 半径要设置得足够大，超过最大的扇形，让光晕散开
        .attr("r", maxRadius + 150)
        // 填充刚才定义的渐变
        .attr("fill", "url(#center-light-glow)")
        // 不要有描边
        .attr("stroke", "none")
        // 确保它不会阻挡鼠标交互（虽然在底层通常不会，但加个保险）
        .style("pointer-events", "none");

    // 1. 绘制背景网格 (白色虚线同心圆)
    const gridTicks = radiusScale.ticks(5).slice(1);
    // 向外追加 2 个刻度 (超出最长扇形)
    // gridTicks.push(lastTick + tickStep);
    // gridTicks.push(lastTick + tickStep * 2);
    const gridG = chartG.append("g").attr("class", "grid-circles");
    gridG.selectAll("circle")
        .data(gridTicks)
        .join("circle")
        .attr("r", d => radiusScale(d))
        .attr("fill", "none")
        .attr("stroke", "#ccc")
        .attr("stroke-dasharray", "3,3")
        .style("opacity", 0.5);
    gridG.selectAll("text")
        .data(gridTicks)
        .join("text")
        .each(function (d) {
            // 沿着 targetAngle 方向放置 label
            const r = radiusScale(d);
            const x = Math.sin(targetAngle) * r;
            const y = -Math.cos(targetAngle) * r;
            d3.select(this).attr("x", x).attr("y", y);
        })
        .attr("dy", "0.35em")
        .text(d => d)
        .attr("text-anchor", "middle")
        .style("font-size", "12px")
        .style("fill", "#999")
        .style("stroke", background_color)
        .style("stroke-width", "3px")
        .style("paint-order", "stroke");

    // 2. 弧生成器
    const overlap = 0.001; // 相邻扇形交叠，减小缝隙
    const arc = d3.arc()
        .innerRadius(innerRadius)
        .outerRadius(d => radiusScale(d[valueKey]))
        .startAngle((d, i) => angleScale(i) + rotationOffset - overlap) // 应用旋转偏移
        .endAngle((d, i) => angleScale(i) + rotationOffset + angleScale.bandwidth() + overlap)
        .padAngle(0)
        .padRadius(0);

    // 3. 绘制扇形 (使用分层叠加法)
    // 创建分组，每个扇形一个组
    const sectorGroups = chartG.selectAll("g.sector-group")
        .data(sortedData)
        .join("g")
        .attr("class", "sector-group");
    // --- 第二层：主体填充层 (Fill Layer) ---
    // 作用：利用径向渐变，实现“远离圆心越亮”的效果
    sectorGroups.append("path")
        .attr("d", (d, i) => arc(d, i))
        .attr("fill", (d, i) => `url(#sector-gradient-${i})`) // 引用径向渐变
        .attr("stroke", "none"); // 这一层不需要描边
    // --- 第一层：光晕层 (Glow Layer) ---
    // 作用：利用模糊的描边，实现“靠近边框越亮”的效果
    // 这一层只画描边，不填充，且使用了模糊滤镜
    sectorGroups.append("path")
        .attr("d", (d, i) => arc(d, i))
        .attr("fill", "none") // 不填充
        .attr("stroke", d => colorScale(d[categoryKey])) // 使用扇形本来的颜色
        .attr("stroke-width", "12px") // 描边要厚，这样模糊后才有光晕感
        .attr("stroke-opacity", 0.6) // 半透明，避免太刺眼
        .style("filter", "url(#glow-blur)") // 应用高斯模糊
        .style("mix-blend-mode", "screen"); // (可选) 让光效叠加更自然
    // --- 第三层：骨架轮廓层 (Border Layer) ---
    // 作用：绘制清晰锐利的分割线 (亮青色/白色)，界定形状
    sectorGroups.append("path")
        .attr("d", (d, i) => arc(d, i))
        .attr("fill", "none")
        .attr("stroke", "rgba(180, 230, 255, 0.5)") // 亮青白色
        .attr("stroke-width", "2px") // 细线
        .style("opacity", 0.8);

    // ============================================================
    // 4. 添加数值标签 & 单飞组标签
    // ============================================================

    // 1. 预计算组大小
    const groupSizes = d3.rollup(sortedData, v => v.length, d => d[groupKey]);

    // 创建一个专门存放数值标签的层，以免跟扇形混淆
    const labelLayer = chartG.append("g").attr("class", "value-labels-layer");

    labelLayer.selectAll(".sector-value-label")
        .data(sortedData)
        .join("g")
        .each(function (d, i) {
            // --- A. 基础计算 ---
            const bandwidth = angleScale.bandwidth();
            const startAngle = angleScale(i) + rotationOffset;
            const endAngle = startAngle + bandwidth;

            // 计算中心角度，判断上下半圆
            const midAng = (startAngle + endAngle) / 2;
            let normalizedMid = midAng % (2 * Math.PI);
            if (normalizedMid < 0) normalizedMid += 2 * Math.PI;
            const isLower = normalizedMid > Math.PI / 2 && normalizedMid < 3 * Math.PI / 2;

            // --- B. 计算路径半径 (位于扇形外部) ---
            const rVal = radiusScale(d[valueKey]);
            const textPadding = 8; // 文字距离扇形边缘的间隙
            const pathRadius = rVal + textPadding;

            // 1. 计算这段弧的物理长度 (可用空间)
            // Arc Length = Radius * Angle (radians)
            // 我们保留 10% 的安全边距，不用满整个弧长
            const availableArcLength = pathRadius * bandwidth * 0.9;

            // 2. 获取文字内容长度
            const textStr = String(d[valueKey]); // 转字符串
            const textLen = textStr.length;

            // 3. 定义字号范围
            const maxFontSize = 30; // 默认最大字号
            const minFontSize = 8;  // 最小字号 (再小就看不清了)

            // 4. 估算所需字号
            // 经验公式：Arial Bold 字体，平均字符宽度约为字号的 0.6 倍
            // Width = Length * Size * 0.6  ==>  Size = Width / (Length * 0.6)
            let adaptiveFontSize = availableArcLength / (textLen * 0.6);

            // 5. 限制范围
            adaptiveFontSize = Math.min(maxFontSize, adaptiveFontSize);
            adaptiveFontSize = Math.max(minFontSize, adaptiveFontSize);

            // --- C. 构建 SVG 路径 (用于 textPath) ---
            // 上半圆：Start -> End (顺时针)
            // 下半圆：End -> Start (逆时针)

            const xStart = Math.sin(startAngle) * pathRadius;
            const yStart = -Math.cos(startAngle) * pathRadius;
            const xEnd = Math.sin(endAngle) * pathRadius;
            const yEnd = -Math.cos(endAngle) * pathRadius;

            let pathD;
            if (isLower) {
                // 逆时针
                pathD = `M ${xEnd},${yEnd} A ${pathRadius},${pathRadius} 0 0,0 ${xStart},${yStart}`;
            } else {
                // 顺时针
                pathD = `M ${xStart},${yStart} A ${pathRadius},${pathRadius} 0 0,1 ${xEnd},${yEnd}`;
            }

            // 生成唯一的 ID
            const valPathId = `val-path-${i}`;
            const g = d3.select(this);

            // 1. 添加不可见路径
            g.append("path")
                .attr("id", valPathId)
                .attr("d", pathD)
                .attr("fill", "none")
                .attr("stroke", "none");

            // 2. 添加数值文字
            g.append("text")
                .attr("dy", isLower ? "0.6em" : "0em") // 微调垂直位置，保证不压线
                .style("font-family", "Arial")
                .style("font-size", `${adaptiveFontSize}px`) // 数值可以稍微大一点
                .style("font-weight", "bold")
                .style("fill", colorScale(d[categoryKey])) // 使用扇形同色
                .style("text-shadow", "0 0 3px #000") // 黑色描边保证清晰
                .append("textPath")
                .attr("href", `#${valPathId}`)
                .attr("startOffset", isLower ? "5%" : "95%")
                .attr("text-anchor", isLower ? "start" : "end")
                .text(d[valueKey]);


            // --- D. 处理单飞组 (Size === 1) ---
            // 如果这个组只有一个扇形，我们需要在数值的【更外层】再画一条弧线写名字
            const size = groupSizes.get(d[groupKey]);
            if (size === 1) {
                const groupPadding = 16;
                const groupPathRadius = pathRadius + groupPadding;
                const iconSize = 24; // 图标大小
                // 1. 定义间距
                // 我们不使用 midAng (中间)，而是使用 startAngle (起始边)
                // 这样图标就在扇形的“顺时针头部”，而数值通常在“尾部”或“中间”
                const edgePaddingPx = 15; // 距离扇形起始边的像素距离
                const anglePerPixel = 1 / groupPathRadius; // 弧度/像素 转换系数
                // 2. 计算图标的中心角度
                // 目标角度 = 起始角 + (边距 + 半个图标宽) * 系数
                const offsetAngle = (edgePaddingPx + iconSize / 2) * anglePerPixel;
                const iconAngle = startAngle + offsetAngle;
                // 3. 计算坐标 (极坐标 -> 直角坐标)
                const imgX = Math.sin(iconAngle) * groupPathRadius;
                const imgY = -Math.cos(iconAngle) * groupPathRadius;
                // 4. 计算旋转 (让图标头朝外，沿着半径方向)
                const rotateDeg = iconAngle * 180 / Math.PI;
                const iconG = g.append("g")
                    .attr("transform", `translate(${imgX}, ${imgY}) rotate(${rotateDeg})`);
                iconG.append("circle")
                    .attr("r", iconSize / 2)
                    .attr("fill", iconFillMap[d[groupKey]])
                    .attr("stroke", colorScale(d[categoryKey]))
                    .attr("stroke-width", 2);
            }
        });

    // ============================================================
    // 5. 绘制 X 轴类别标签 (Category Labels)
    // ============================================================
    chartG.selectAll(".category-marker")
        .data(sortedData)
        .join("text")
        .text(d => d[categoryKey])
        .each(function (d, i) {
            // 计算扇形角度参数
            const bandwidth = angleScale.bandwidth();
            const startAngle = angleScale(i) + rotationOffset;
            const endAngle = startAngle + bandwidth;

            const centerAngle = startAngle + bandwidth / 2;
            let normalizedAng = centerAngle % (2 * Math.PI);
            if (normalizedAng < 0) normalizedAng += 2 * Math.PI;
            const isRightSide = normalizedAng < Math.PI;

            // --- 1. 确定定位角度 (沿着下边缘) ---
            let targetAngle;
            if (isRightSide) {
                targetAngle = endAngle ; // 右侧用结束角
            } else {
                targetAngle = startAngle; // 左侧用起始角
            }

            // --- 2. 计算坐标 (定位在扇形的最外沿) ---
            const rVal = radiusScale(d[valueKey]);
            const textRadius = Math.max(rVal - 25, 60);
            const x = Math.sin(targetAngle) * textRadius;
            const y = -Math.cos(targetAngle) * textRadius;

            // --- 3. 计算旋转 ---
            let rotateDeg = targetAngle * 180 / Math.PI;
            if (isRightSide) {rotateDeg -= 90; // 文字从内向外
            } else {
                rotateDeg += 90; // 文字从外向内
            }

            // A. 获取文字内容和长度
            const textContent = d[categoryKey];
            const textLength = textContent.length; // 字符数

            // B. 计算物理弧长 (Arc Width)
            // 乘以 0.9 是为了留出 10% 的左右安全边距
            const arcWidth = textRadius * bandwidth * 0.9;

            // C. 估算每个字符的宽度比例 (Width/FontSize)
            // Arial Bold 字体，英文平均宽度约为字号的 0.6 倍，中文约为 1.0 倍
            // 简单判断：如果包含中文，系数设大一点
            const hasChinese = /[\u4e00-\u9fa5]/.test(textContent);
            const charRatio = hasChinese ? 1.0 : 0.6;

            // D. 反推合适的字号
            // 公式：ArcWidth = FontSize * charRatio * textLength
            // 所以：FontSize = ArcWidth / (textLength * charRatio)
            let calculatedSize = arcWidth / (Math.sqrt(textLength) * charRatio) * 0.6;

            // E. 限制最大最小值
            const maxFontSize = 30; // 最大字号
            const minFontSize = 9;  // 最小字号

            let finalFontSize = Math.min(maxFontSize, calculatedSize);
            finalFontSize = Math.max(minFontSize, finalFontSize);

            const el = d3.select(this);
            const padding = 0; // 文字距离外弧的间距

            el.attr("transform", `translate(${x}, ${y}) rotate(${rotateDeg})`)
                // 垂直偏移：向上提一点，避开下边缘线
                .attr("dy", "-0.3em")
                .style("font-family", "Arial")
                .style("font-size", `${finalFontSize}px`)
                .style("font-weight", "bold")
                .style("fill", "rgba(255,255,255, 0.9)")
                .style("text-shadow", "0 0 4px rgba(0,0,0,0.8)")
                .style("pointer-events", "none");

            // --- 4. 关键修改：对齐方式与间距 ---
            if (isRightSide) {
                // 右侧：从内向外写。为了贴着外弧，设为【右对齐】
                // dx 为负数，表示往回(圆心方向)缩一点
                el.attr("text-anchor", "end")
                    .attr("dx", `-${padding}px`);
            } else {
                // 左侧：从外向内写。为了贴着外弧，设为【左对齐】
                // dx 为正数，表示顺着文字方向(圆心方向)推一点
                el.attr("text-anchor", "start")
                    .attr("dx", `${padding}px`);
            }
        });

    // ============================================================
    // 5.5. 绘制分组外环 (Group Arcs)
    // ============================================================

    // 1. 按 Group 聚合数据，我们需要知道每组的起始和结束索引
    // 因为 sortedData 已经排好序了，同组数据是连续的
    const groupMap = d3.group(sortedData, d => d[groupKey]);
    const groupsArray = Array.from(groupMap); // 转成数组方便遍历 [key, values]

    // 创建一个层专门放环，放在最下面或者最上面都可以，这里放在 Label 下面
    const arcsLayer = chartG.append("g").attr("class", "group-arcs-layer");
    groupsArray.forEach((entry, groupIndex) => {
        const gName = entry[0];
        const gData = entry[1];
        // 规则：如果这个 Group 只有一个，不画环 (已在 label 处理)
        if (gData.length <= 1) return;
        // 2. 找到该组在 sortedData 中的索引范围
        // 简单方法：映射 data 得到 index 数组
        const indices = gData.map(d => sortedData.indexOf(d));
        const minIdx = Math.min(...indices);
        const maxIdx = Math.max(...indices);
        // 3. 计算环的角度范围
        const startAngle = angleScale(minIdx) + rotationOffset;
        // 结束角度 = 最后一个扇形的起始角 + 带宽
        const endAngle = angleScale(maxIdx) + angleScale.bandwidth() + rotationOffset;
        // 4. 计算环的半径 (防遮挡 + 错落)
        // 4.1 找出该组内最长的扇形半径
        const maxRInGroup = d3.max(gData, d => radiusScale(d[valueKey]));
        // 4.2 基础半径：必须比组内最长扇形大，还要避开数值标签
        // 之前数值标签大概推了 15px + 文字宽度，我们这里推得更远一点
        let arcRadius = maxRInGroup + 50;
        // 4.3 错落高度：相邻的环不能在同一高度
        // 我们用 groupIndex 的奇偶性来决定高度
        // 偶数层低一点，奇数层高一点 (相差 15px)
        if (groupIndex % 2 !== 0) {
            arcRadius += 18;
        }
        // 定义环的厚度和圆角
        const ringThickness = 40; // 环的厚度
        const ringCornerRadius = 15; // 圆角半径 (厚度的一半 = 半圆头)
        // 5. 绘制弧线路径 (Path)
        // 定义一个不可见的路径用于文字依附 (TextPath)，或者直接画可视化的线
        // 1. 根据 groupIndex 循环从调色板取色
        const rawColor = myCustomPalette[groupIndex % myCustomPalette.length];
        // 2. 生成浅色填充 (Fill): 极低的不透明度
        // 使用 d3.color() 解析颜色并设置 opacity
        const whitishColor = d3.interpolateRgb(rawColor, "#ffffff")(0.6);
        const fillColor = d3.color(whitishColor).copy({ opacity: 0.3 });
        // 3. 生成稍深一点的描边 (Stroke): 中等不透明度
        const strokeColor = d3.color(whitishColor).copy({ opacity: 0.3 });
        const arcGen = d3.arc()
            .innerRadius(arcRadius)
            .outerRadius(arcRadius + ringThickness)
            .startAngle(startAngle)
            .endAngle(endAngle)
            .padAngle(0)
            .cornerRadius(ringCornerRadius);
        // Layer 1: 光晕层 (Glow Layer)
        // 放在下面，只描边，加模糊
        arcsLayer.append("path")
            .attr("class", "arc-glow")
            .attr("d", arcGen())
            .attr("fill", "none") // 不要填充，否则整个环都会糊掉
            .attr("stroke", strokeColor) // 使用高亮的描边色
            .attr("stroke-width", "20px") // 描边要厚，模糊开来才好看
            .style("filter", "url(#glow-blur)") // 复用之前定义的高斯模糊滤镜
            .style("opacity", 0.8); // 稍微透明一点，不要太刺眼
        // Layer 2: 实体层 (Body Layer)
        // 放在上面，清晰的磨砂玻璃质感
        arcsLayer.append("path")
            .attr("class", "arc-body")
            .attr("d", arcGen())
            .attr("fill", fillColor)
            .attr("stroke", strokeColor)
            .attr("stroke-width", "1.5px");
        // 绘制可见的亮色弧线
        arcsLayer.append("path")
            .attr("d", arcGen())
            .attr("fill", fillColor) // 半透明白色填充 (环状条)
            .attr("stroke", strokeColor) // 稍微深一点的描边
            .attr("stroke-width", "1px");

        // 6. 绘制环上的文字 (Group Name)
        // 使用 textPath 让文字弯曲贴合弧线
        // 判断是否在下半圆，决定文字是否要翻转
        // 计算弧线中心角度
        // --- 1. 基础参数计算 ---
        const textR = arcRadius + (ringThickness / 2);

        // 判断是否在下半圆 (用于决定路径方向和对齐方式)
        // 计算扇形的中间角度来判断整体位置
        const midAngle = (startAngle + endAngle) / 2;
        let normalizedMid = midAngle % (2 * Math.PI);
        if (normalizedMid < 0) normalizedMid += 2 * Math.PI;
        const isLower = normalizedMid > Math.PI / 2 && normalizedMid < 3 * Math.PI / 2;

        // --- 2. 定义间距参数 (像素 & 角度) ---
        const iconSize = 24;
        const edgePaddingPx = 15; // 顺时针起始处的留白间距 (像素)
        const textIconGapPx = 8;  // 图标和文字之间的间距 (像素)

        // 计算 "像素" 对应的 "角度"
        // 公式: 角度 = 弧长 / 半径
        const anglePerPixel = 1 / textR;

        // --- 3. 计算图片位置 (始终在 startAngle 的顺时针方向) ---
        // 图片中心角度 = 起始角 + (边缘留白 + 半个图标) * 像素转角度系数
        // 注意：startAngle 永远是扇形的顺时针起始边 (数值较小的边)
        const imgCenterOffsetAngle = (edgePaddingPx + iconSize / 2) * anglePerPixel;
        const finalImgAngle = startAngle + imgCenterOffsetAngle;

        const imgX = Math.sin(finalImgAngle) * textR;
        const imgY = -Math.cos(finalImgAngle) * textR;
        const rotateDeg = finalImgAngle * 180 / Math.PI;

        // --- 4. 绘制图片/圆点 ---
        const iconG = arcsLayer.append("g")
            .attr("transform", `translate(${imgX}, ${imgY}) rotate(${rotateDeg})`);
        iconG.append("circle")
            .attr("r", iconSize / 2)
            // 直接读取预定义的材质！
            .attr("fill", iconFillMap[gName])
            .attr("stroke", strokeColor)
            .attr("stroke-width", 1);

        // --- 5. 绘制文字 (textPath) ---

        // 计算文字应该开始的 "像素偏移量" (从 startAng 开始算)
        // 偏移 = 边缘留白 + 图标完整宽度 + 图文间距
        const totalOffsetPx = edgePaddingPx + iconSize + textIconGapPx;

        // 构建路径
        const xStart = Math.sin(startAngle) * textR;
        const yStart = -Math.cos(startAngle) * textR;
        const xEnd = Math.sin(endAngle) * textR;
        const yEnd = -Math.cos(endAngle) * textR;

        let pathD;
        let startOffsetVal;
        let anchorVal;
        let dxVal;

        if (isLower) {
            // 【下半圆】：路径从 End(左) 画到 Start(右)，为了让文字直立
            // 路径方向：左 -> 右
            // 我们希望文字出现在 Start(右) 附近
            // 所以文字位于路径的 "终点" 附近
            pathD = `M ${xEnd},${yEnd} A ${textR},${textR} 0 0,0 ${xStart},${yStart}`;

            // 设置对齐
            startOffsetVal = "100%"; // 从路径末尾开始
            dxVal = `-${totalOffsetPx}px`; // 往回退 (向左退)，留出空间给图片
            anchorVal = "end"; // 文字末尾对齐锚点

        } else {
            // 【上半圆】：路径从 Start(左) 画到 End(右)
            // 路径方向：左 -> 右
            // 我们希望文字出现在 Start(左) 附近
            // 所以文字位于路径的 "起点" 附近
            pathD = `M ${xStart},${yStart} A ${textR},${textR} 0 0,1 ${xEnd},${yEnd}`;

            // 设置对齐
            startOffsetVal = `${totalOffsetPx}px`; // 从起点往前走一点，跳过图片
            dxVal = "0px";
            anchorVal = "start"; // 文字开头对齐锚点
        }

        const pathId = `group-text-path-${groupIndex}`;
        arcsLayer.append("path")
            .attr("id", pathId)
            .attr("d", pathD)
            .attr("fill", "none", "stroke", "none"); // 不可见路径

        const textLabel = arcsLayer.append("text")
            .style("font-family", "Arial")
            .style("font-size", "20px")
            .style("font-weight", "bold")
            .style("fill", "#ffffff")
            .style("text-shadow", "0 0 5px rgba(0,0,0,0.5)")
            .style("letter-spacing", "2px")
            .attr("dx", dxVal); // 应用偏移

        textLabel.append("textPath")
            .attr("href", `#${pathId}`)
            .attr("startOffset", startOffsetVal)
            .attr("text-anchor", anchorVal)
            .attr("side", isLower ? "left" : "right") // 修正基线方向
            .attr("dominant-baseline", "middle")
            .text(gName);
    });

    // --- C. 图例 (Legend) ---
    // 1. 为图例单独定义一个稍微柔和一点的高斯模糊滤镜
    const legendFilter = defs.append("filter")
        .attr("id", "legend-glow")
        .attr("x", "-50%").attr("y", "-50%")
        .attr("width", "200%").attr("height", "200%");
    legendFilter.append("feGaussianBlur")
        .attr("stdDeviation", "4") // 模糊程度，比主图小一点，适合图标
        .attr("result", "coloredBlur");
    // 放在左边区域
    const legendG = svg.append("g")
        .attr("transform", `translate(${margin.left}, ${height - 700})`);
    allCategories.forEach((cat, i) => {
        const row = i;
        const g = legendG.append("g")
            .attr("transform", `translate(0, ${row * 60})`); // 稍微增加行高，留出光晕空间

        // --- 1. 定义每个图例的独立渐变 ---
        const legendGradId = `legend-grad-${i}`;
        const legendGrad = defs.append("radialGradient")
            .attr("id", legendGradId)
            .attr("cx", "50%")
            .attr("cy", "50%")
            .attr("r", "50%");
        // 渐变中心：背景色 (模拟扇形根部)
        legendGrad.append("stop")
            .attr("offset", "0%")
            .attr("stop-color", background_color)
            .attr("stop-opacity", 1);
        // 渐变边缘：分类颜色
        legendGrad.append("stop")
            .attr("offset", "100%")
            .attr("stop-color", colorScale(cat))
            .attr("stop-opacity", 1);
        // --- 2. 绘制最外层的高斯模糊光晕 (Outer Glow) ---
        g.append("circle")
            .attr("r", 20)      // 半径
            .attr("cx", 25)     // 圆心 X (留出左边距)
            .attr("cy", 20)     // 圆心 Y
            .attr("fill", "none")
            .attr("stroke", colorScale(cat))
            .attr("stroke-width", 4) // 描边作为发光源
            .style("filter", "url(#legend-glow)") // 应用模糊
            .style("opacity", 0.6);
        // --- 3. 绘制主体圆形 (带渐变) ---
        g.append("circle")
            .attr("r", 20)
            .attr("cx", 25)
            .attr("cy", 20)
            .attr("fill", `url(#${legendGradId})`) // 使用上面定义的渐变
            .attr("stroke", colorScale(cat))       // 稍微加个实边框增加清晰度
            .attr("stroke-width", "1px");
        // --- 4. 绘制文字 (增加间距) ---
        g.append("text")
            .text(cat)
            .attr("x", 65)      // 原来是40，现在改大，留出与圆形的空隙
            .attr("y", 25)      // 文字垂直居中对齐 (圆心在20，文字基线在30左右比较居中)
            .style("font-family", typeLabel.font_family)
            .style("font-size", "25px") // 保持你设定的大字体
            .style("font-weight", "bold")
            .style("fill", text_color)
            .style("alignment-baseline", "middle"); // 辅助垂直居中
    });

    // --- D. 侧边栏 (Insights) ---

    // 1. 定义侧边栏的布局区域
    // X: 从 0 开始 (贴着左边)
    // Y: 放在 headerHeight 附近 (比标题稍微靠下一点的位置)
    const sidebarAreaX = -100
    const sidebarAreaY = 100; // 你可以调整这个数值来控制垂直位置
    const sidebarAreaW = 400;      // 使用之前定义的 380px 宽度
    const sidebarAreaH = 400;               // 限制最大高度
    const insightG = svg.append("g")
        .attr("transform", `translate(${sidebarAreaX}, ${sidebarAreaY})`);

    // 3. Insight 正文 (应用自动缩放逻辑)
    if (insights) {
        // 定义正文的起始 Y (在标题下面)
        const textStartY = 50;
        const textAreaH = sidebarAreaH - textStartY; // 剩下的高度给正文

        const insightText = insightG.append("text")
            .attr("y", textStartY)
            .attr("text-anchor", "start") // 居中对齐
            .attr("dominant-baseline", "hanging")
            .style("font-family", typeAnnot.font_family)
            .style("font-style", "italic")
            .style("fill", text_color)
            .style("opacity", 0.9);

        // 调用自适应函数
        fitTextByWordCount(
            insightText,       // 目标元素
            insights,          // 文本内容
            5,                 // 【参数】每行单词数 (侧边栏较窄，建议 5-7 个词)
            sidebarAreaW - 40, // 【参数】最大宽度 (减去 40px 左右边距)
            textAreaH,         // 【参数】最大高度
            30,                // 【参数】最大字号 (正文不宜过大)
            sidebarAreaW / 2   // 【参数】X坐标中心点 (侧边栏的一半)
        );
    }
    // --- E. 底部来源 ---
    svg.append("text")
        .text("Source: Data provided via JSON")
        .attr("x", width - 50)
        .attr("y", height - 30)
        .attr("text-anchor", "end")
        .style("font-family", "Arial")
        .style("font-size", "20px")
        .style("fill", "#888");
}

// ---------------------------------------------------------
// 辅助函数：左对齐换行
// ---------------------------------------------------------
function wrapText(textSelection, width) {
    textSelection.each(function () {
        const text = d3.select(this);
        const words = text.text().split(/\s+/).reverse();
        let word, line = [], lineNumber = 0, lineHeight = 1.2, y = text.attr("y"), dy = 0;
        const x = text.attr("x") || 0;

        let tspan = text.text(null).append("tspan").attr("x", x).attr("y", y).attr("dy", dy + "em");

        while (word = words.pop()) {
            line.push(word);
            tspan.text(line.join(" "));
            if (tspan.node().getComputedTextLength() > width) {
                line.pop();
                tspan.text(line.join(" "));
                line = [word];
                tspan = text.append("tspan").attr("x", x).attr("y", y).attr("dy", ++lineNumber * lineHeight + dy + "em").text(word);
            }
        }
    });
}

// ---------------------------------------------------------
// 辅助函数：居中对齐换行
// ---------------------------------------------------------
function wrapTextCentered(textSelection, width) {
    let totalHeight = 0;
    textSelection.each(function () {
        const text = d3.select(this);
        const words = text.text().split(/\s+/).reverse();
        let word, line = [], lineNumber = 0, lineHeight = 1.4;
        const y = text.attr("y") || 0;

        text.text(null);

        let tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", 0);

        while (word = words.pop()) {
            line.push(word);
            tspan.text(line.join(" "));
            if (tspan.node().getComputedTextLength() > width) {
                line.pop();
                tspan.text(line.join(" "));
                line = [word];
                tspan = text.append("tspan").attr("x", 0).attr("y", y).attr("dy", ++lineNumber * lineHeight + "em").text(word);
            }
        }
        totalHeight = (lineNumber + 1) * 20;
    });
    return totalHeight;
}


// ---------------------------------------------------------
// 辅助函数：根据单词数自适应字号
// ---------------------------------------------------------
function fitTextByWordCount(textNode, textString, wordsPerLine, maxWidth, maxHeight, maxFontSize, xPos = 0) {
    // 1. 切分单词
    const words = textString.split(/\s+/);
    const lines = [];
    for (let i = 0; i < words.length; i += wordsPerLine) {
        lines.push(words.slice(i, i + wordsPerLine).join(" "));
    }

    // 2. 迭代寻找字号
    let fontSize = maxFontSize;
    const minFontSize = 10;

    while (fontSize >= minFontSize) {
        textNode.style("font-size", `${fontSize}px`);
        textNode.text(null);

        // 设置父级 text 的 x 位置 (关键，确保整体锚点正确)
        textNode.attr("x", xPos);

        const lineHeight = fontSize * 1.2;

        lines.forEach((line, i) => {
            textNode.append("tspan")
                .attr("x", xPos) // 【修改】每行都必须对齐到传入的 xPos (无论是0还是中心点)
                .attr("dy", i === 0 ? 0 : lineHeight)
                .text(line);
        });

        const bbox = textNode.node().getBBox();

        if (bbox.height <= maxHeight && bbox.width <= maxWidth) {
            return bbox.height;
        }

        fontSize -= 2;
    }
    return maxHeight;
}
