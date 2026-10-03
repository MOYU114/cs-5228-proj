"""Build the reviewable Chinese report from saved EDA results, not hand-entered metrics."""
from pathlib import Path
import json
import pandas as pd

def table(df, decimals=2):
    def cell(x):
        if pd.isna(x): return '—'
        if isinstance(x,float):
            return f'{int(x):,}' if x.is_integer() else f'{x:,.{decimals}f}'
        if isinstance(x,int): return f'{x:,}'
        return str(x).replace('|',' / ').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(map(str,df.columns))+' |','| '+' | '.join(['---']*len(df.columns))+' |']+
                     ['| '+' | '.join(cell(x) for x in row)+' |' for row in df.itertuples(index=False,name=None)])

def write_report(root: Path):
    output=root/'experiments'/'eda'; core=output/'core'
    read=lambda n: pd.read_csv(core/f'{n}.csv')
    s=json.loads((core/'summary.json').read_text()); r=s['rent']
    manifest=json.loads((output/'run_manifest.json').read_text())
    p=read('field_overview'); dup=read('duplicates'); drift=read('feature_drift')
    field_rows=[]
    units={'RENT_APPROVAL_DATE':'批准月份，YYYY-MM','TOWN':'城镇','BLOCK':'楼号字符串（不独立唯一）','STREET':'街道字符串',
           'FLAT_TYPE':'户型','FLAT_MODEL':'设计类别，勿与户型混同','FLOOR_AREA_SQM':'m²','FURNISHED':'配置类别','LEASE_COMMENCE_DATE':'租约起始年份（非完整日期）',
           'FEE':'费用，当前全为零；具体定义待确认','MONTHLY_RENT':'SGD/月，只有 train 有'}
    for col in p.field.drop_duplicates():
        a=p[(p.split=='train')&(p.field==col)].iloc[0]
        b=p[(p.split=='test')&(p.field==col)]
        value=f'{a["min"]}–{a["max"]}' if col in ['FLOOR_AREA_SQM','LEASE_COMMENCE_DATE','MONTHLY_RENT','RENT_APPROVAL_DATE'] else f'{int(a["unique"]):,} 个原始值'
        if col=='FURNISHED': value='yes（常量）'
        if col=='FEE': value='0.0（常量）'
        field_rows.append([col,units[col],a['dtype'],value,'—' if b.empty else int(b.iloc[0]['unique']),int(a.missing),'—' if b.empty else int(b.iloc[0].missing)])
    field_table=table(pd.DataFrame(field_rows,columns=['字段','意义/单位','train 读取类型','train 范围/类别','test 唯一值数','train 缺失','test 缺失']))
    flat=read('rent_by_flat_type'); flat=flat.set_index('FLAT_TYPE').reindex(['1-room','2-room','3-room','4-room','5-room','executive']).reset_index()
    town=read('rent_by_town').sort_values('median')
    month=read('rent_by_month'); first,last=month.iloc[0],month.iloc[-1]
    years=read('rent_by_year');years['YEAR']=years.YEAR.astype(str).replace({'2025':'2025 Q1（非全年）'})
    renames={'n':'样本数','mean':'平均租金','median':'租金中位数','q25':'P25','q75':'P75'}
    summary_table=lambda d,key: table(d[[key,'n','mean','median','q25','q75']].rename(columns=renames))
    ds=dup[dup.definition.isin(['all_available_fields','X_only','block_month'])][['split','form','definition','repeated_groups','rows_in_repeated_groups','excess_rows']]
    ft_raw=drift[(drift.form=='raw')&(drift.reference=='all_train')&(drift.field=='FLAT_TYPE')].iloc[0].value
    cats=drift[(drift.form=='normalized')&(drift.metric=='TVD')].pivot(index='field',columns='reference',values='value').reset_index()
    nums=drift[(drift.reference=='all_train')&(drift.metric=='standardized_mean_difference')][['field','train_mean','test_mean','value','test_outside_train_range']]
    flags=read('anomaly_flags');iqr=flags[flags.rule.str.contains('IQR')]
    neg=read('train_negative_age').iloc[0]
    cv=s['block_coverage'][0]; recent_cv=s['block_coverage'][1]
    sw=s['suggested_window_counts']
    mapping=read('category_normalization_map')
    fm=mapping[(mapping.field=='FLAT_TYPE')&(mapping.split=='train')][['raw','normalized','n']].sort_values(['normalized','raw'])
    changes=[]
    for split in ['train','test']:
        for col,vals in s['normalization'][split].items():
            if vals['changed_rows']: changes.append([split,col,vals['raw_unique'],vals['normalized_unique'],vals['changed_rows']])
    change_table=table(pd.DataFrame(changes,columns=['split','字段','原始类别数','规范化类别数','改变写法的行数']))
    numeric=read('numeric_summary'); rng=[]
    for col in ['FLOOR_AREA_SQM','LEASE_COMMENCE_DATE']:
        for split in ['train','test']:
            a=numeric[(numeric.field==col)&(numeric.split==split)].iloc[0]
            values=[a[k] for k in ['min','25%','50%','75%','max']]
            if col=='LEASE_COMMENCE_DATE': values=[str(int(v)) for v in values]
            rng.append([split,col,*values])
    text=f'''# 阶段一：EDA 与数据质量审计

本报告由 `python notebooks/run_eda.py` 根据本地 CSV 和运行结果生成，角色 A，提交团队审阅。范围仅为 EDA、质量审计及后续实验假设；没有训练模型、比较模型成绩、固定候选方案数量、生成预测或提交 Kaggle。没有改动原始 CSV、`proposal/main.md`、`proposal/slides.md` 或最终报告。

## 1. 数据、口径与复现

分析依据为 README、课程说明及三份团队指南。数字取自本次实际运行的 [核心汇总](../experiments/eda/core/summary.json)、[辅助汇总](../experiments/eda/auxiliary/summary.json) 和同目录明细表；不是从课程描述抄录。训练集 {s['rows']['train']:,} 行、{s['columns']['train']} 列，测试集 {s['rows']['test']:,} 行、{s['columns']['test']} 列。训练范围 {s['dates']['train']['min']}–{s['dates']['train']['max']}（{s['dates']['train']['months']} 个有记录月份），测试范围 {s['dates']['test']['min']}–{s['dates']['test']['max']}（{s['dates']['test']['months']} 个有记录月份）。测试文件没有 `MONTHLY_RENT`，代码对此有断言。

所有租金统计只用 train 的目标值。测试特征仅用于用户要求的分布、覆盖和连接审计，没有参与任何模型、编码器、填补器或阈值的拟合。全期间辅助表仅用于核查字段和时间边界，没有用未来经济观测构造历史或测试特征。内存中的规范化副本保留全部观测，不去重、不剔除尾部、不插补；输出 CSV 均在 `experiments/eda/`。

复现（从项目根目录，推荐 Python {manifest['python']}，依赖固定见 [requirements-eda.txt](../notebooks/requirements-eda.txt)）：

```bash
python3 -m venv .venv-eda
.venv-eda/bin/python -m pip install -r notebooks/requirements-eda.txt
.venv-eda/bin/python notebooks/run_eda.py
```

本机实跑使用 Codex bundled Python，绘图依赖安装在临时目录；可在当前环境复跑：

```bash
PYTHONPATH=/private/tmp/cs5228-eda-deps /Users/moyu114/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 notebooks/run_eda.py
```

临时目录不保证跨会话存在；干净环境请使用上面的 venv 命令。依赖实跑版本为 pandas {manifest['packages']['pandas']}、NumPy {manifest['packages']['numpy']}、Matplotlib {manifest['packages']['matplotlib']}；不依赖 sklearn、SciPy、网络 API 或外部数据。完整分析使用全部记录，无随机抽样。[运行清单](../experiments/eda/run_manifest.json) 保存输入和代码 SHA-256、Python/包版本、受保护文件哈希及命令。脚本运行前后核对原始 CSV 与受保护文档哈希一致；源码运行时写入的路径限于本阶段交付物。

统计口径：缺失数含 pandas 识别的 NA，另外检查空白和辅助表哨兵值；唯一值数默认不含 NA。重复“多余行”是每个相同组保留一行后的剩余数量，仅作为统计定义，未实际删除。租金单位 SGD/月，面积 m²。`AGE_YEARS = 批准年份 − LEASE_COMMENCE_DATE` 是年度精度的租约楼龄代理，不是精确竣工楼龄。图中 IQR 是观测分布的 P25–P75，不是置信区间。

## 2. 核心字段与质量

{field_table}

两集所有核心字段的 NA 与空白字符串均为 0。月份全部能按 `%Y-%m` 解析，面积和租金均为正且有限，租约年份无非整数值。`BLOCK` 以字符串读取保留字母后缀；单独的楼号不是地址唯一键。原始字符串的最小/最大值仅有字典序意义；完整逐字段类型、范围及常量标记见 [field_overview.csv](../experiments/eda/core/field_overview.csv)，各类别完整频数见 [category_frequencies_raw.csv](../experiments/eda/core/category_frequencies_raw.csv)。

面积与租约年份范围核对：

{table(pd.DataFrame(rng,columns=['split','字段','最小','P25','中位数','P75','最大']))}

`FURNISHED=yes` 与 `FEE=0.0` 在两集均为常量，现有样本无法估计它们与租金的关系。建议派生特征中排除并保留 schema 审计，不解释为“家具或费用没有影响”。FLAT_MODEL 在训练有 {s['normalization']['train']['FLAT_MODEL']['raw_unique']} 类，测试有 {s['normalization']['test']['FLAT_MODEL']['raw_unique']} 类；同名或罕见类别仍需折内处理，类别频数不等于预测效果。

### 2.1 类别写法

{change_table}

保守规则为 Unicode NFKC → 去首尾空白 → 小写 → 连续空白压成一个空格；对 `FLAT_TYPE` 单独用 `^(\\d+)\\s*[- ]\\s*room$` 映射成 `N-room`，`executive` 保持。不要把 FLAT_MODEL 的 `2 room` 与 FLAT_TYPE 混为同一字段，也不要将楼号转整数、删除后缀或凭猜测展开街道缩写。未发现需要模糊匹配才能连接 HDB 的证据。

训练户型频数（未删行）：

{table(fm)}

全部原值→规范值及数量在 [category_normalization_map.csv](../experiments/eda/core/category_normalization_map.csv)；可复用函数位于 [eda_core.py](../notebooks/eda_core.py) 的 `normalized_copy`。该函数只做确定性文本清洗，不学习全数据词表。

### 2.2 异常值与可疑值

{table(iqr.rename(columns={'rule':'仅用于标记的规则','lower':'下界','upper':'上界','n':'界外行数'}))}

IQR 界外定义为严格小于下界或大于上界。上下界只由全训练集的 P25/P75 计算，用于本次质量审计；后续验证若使用异常阈值，必须在每个训练折重新计算，不得沿用全训练阈值处理早期验证。IQR 不是错误判定，当前没有删除这些行。尤其租金尾部可能是真实交易，不能为了降低 RMSE 直接删去验证尾部。

负楼龄只有训练中 {int(flags[(flags.split=='train')&(flags.rule=='negative_age')].iloc[0].n)} 行：零基数据行 `{int(neg.source_row_0based)}`（CSV 含表头第 {int(neg.source_row_0based)+2} 行），`{neg.RENT_APPROVAL_DATE} / {neg.TOWN} / {neg.BLOCK} / {neg.STREET}`，租约年份 {int(neg.LEASE_COMMENCE_DATE)}，年份差为 −1，租金 {int(neg.MONTHLY_RENT):,} SGD/月。IQR 没有识别此语义异常，说明还需要逻辑校验。先保留异常指示；核查批准与起租/租约年份口径后，再决定是否在派生特征中置缺失。不要擅自把年份改成批准年或把年龄截成零。

更多详情见 [anomaly_flags.csv](../experiments/eda/core/anomaly_flags.csv) 和 [rent_extreme_examples.csv](../experiments/eda/core/rent_extreme_examples.csv)；后者是最低/最高租金的示例切片，不能当作穷尽的错误清单。

## 3. MONTHLY_RENT 的分布与关联

总体平均 {r['mean']:,.2f}、标准差 {r['std']:,.2f}、中位数 {r['50%']:,.0f} SGD/月；范围 {r['min']:,.0f}–{r['max']:,.0f}，P1/P99 为 {r['1%']:,.0f}/{r['99%']:,.0f}，P25/P75 为 {r['25%']:,.0f}/{r['75%']:,.0f}。样本偏度 {r['skew']:.3f}，右尾存在但不能因此预设对数目标有效。{r['multiple_50_fraction']:.2%} 的租金是 50 SGD 的整数倍，表明值有明显离散集中；原因及是否为源数据舍入尚未核实。

![训练租金分布](../proposal/figures/eda_rent_distribution.png)

图示保留全范围，直方图每格 100 SGD；累积分布显示尾部只占少量记录。RMSE 对较大残差敏感，后续应报告各租金区间误差；这不是截尾或改变目标函数的效果证据。

### 3.1 月份

![月份覆盖与租金](../proposal/figures/eda_month_coverage_and_rent.png)

观测事实：{first.RENT_APPROVAL_DATE} 平均租金 {first['mean']:,.2f}（n={int(first.n):,}），{last.RENT_APPROVAL_DATE} 为 {last['mean']:,.2f}（n={int(last.n):,}）。跨年的描述性统计如下，单位 SGD/月：

{summary_table(years,'YEAR')}

训练中上升主要发生在较早区段，后期月均仍有波动。2025 行仅为第一季度，不能与全年平均当作同比增幅。月均和中位数同时受成交构成、月份和其他条件影响；图没有调整房源构成，不能作因果解释，也没有外推测试租金。

### 3.2 城镇与户型

![城镇租金](../proposal/figures/eda_rent_by_town.png)

按全训练期中位数排序，低端示例 `{town.iloc[0].TOWN}` 的中位数 {town.iloc[0]['median']:,.0f}（n={int(town.iloc[0].n):,}），高端示例 `{town.iloc[-1].TOWN}` 为 {town.iloc[-1]['median']:,.0f}（n={int(town.iloc[-1].n):,}）SGD/月。图中各城镇 IQR 广泛重叠，样本量不均；完整数值见 [rent_by_town.csv](../experiments/eda/core/rent_by_town.csv)。这是实际出租样本的边际分布，不能视为控制了面积和时间的地段溢价。

![户型和月度租金](../proposal/figures/eda_rent_by_flat_type.png)

{summary_table(flat,'FLAT_TYPE')}

单位 SGD/月。户型越大的组通常具有较高的描述性租金；但面积、年代和区域也随户型变化。右图按户型分层后仍观察到月度变化，因此“总体上涨全由户型比例改变造成”与这些图不一致，但其他构成影响尚未排除。1-room 训练样本仅 {int(flat[flat.FLAT_TYPE=='1-room'].iloc[0].n)} 条，故右侧月线不展示该稀疏组，左侧与总量统计仍保留。各月各户型样本量可查 [rent_by_month_flat_type.csv](../experiments/eda/core/rent_by_month_flat_type.csv)。

### 3.3 面积与楼龄代理

![面积、楼龄与租金](../proposal/figures/eda_rent_area_age.png)

六边形颜色表示格内记录数，对数色标，全训练样本无抽样。租金与面积 Pearson 相关 {r['pearson_area']:.3f}、Spearman 秩相关 {r['spearman_area']:.3f}；与年份差楼龄 Pearson 相关 {r['pearson_age']:.3f}。这些边际相关不足以解释条件效应或证明线性关系，年龄还同时携带批准年和租约起始年代的信息。

面积分箱（m²；左闭右开，最后一箱无上界），租金单位 SGD/月：

{summary_table(read('rent_by_area_band'),'AREA_BAND')}

楼龄代理分箱（年），租金单位 SGD/月：

{summary_table(read('rent_by_age_band'),'AGE_BAND')}

面积组中位数在较大面积段没有按比例持续增长；楼龄组的差异也不是严格单调。可据此提出非线性和交互项实验，但不能确定其预测收益。

## 4. 训练与测试的时间、特征漂移

**事实**：测试共有 {s['dates']['test']['months']} 个日历月份，包含与训练重叠的 2025-03（train {int(month[month.RENT_APPROVAL_DATE=='2025-03'].iloc[0].n):,} 行、test {s['test_rows_overlapping_months']:,} 行）。测试有 {s['test_rows_after_train_max']:,} 行（{s['test_rows_after_train_max']/s['rows']['test']:.2%}）晚于训练最大月份；最远 2026-07 相对 2025-03 的月序差为 {s['horizon_months']}。因此“17 个月份”与“16 个月最大预测间距”同时成立，不能写成全部测试都在训练之后。月份内部没有日期，无法判断重叠月份交易的先后，也不能臆测按行号代表时间。课程说明截至 2026-06，而实际 CSV 到 2026-07，应由团队确认说明差异。

![类别分布比较](../proposal/figures/eda_categorical_shift.png)

用 TVD = 0.5 × Σ|训练类别占比 − 测试类别占比| 描述类别差异，范围 0–1；没有把经验阈值或显著性检验当作模型选择标准。下表为规范化后结果，recent_train 指 2024-10–2025-03 的 {sw['holdout_rows']:,} 行：

{table(cats,4)}

原始 FLAT_TYPE 的全训练 TVD 为 {ft_raw:.4f}，规范化后为 {float(cats[cats.field=='FLAT_TYPE'].iloc[0]['all_train']):.4f}，说明原始表观漂移中有很大一部分是写法差异。规范化后 test 未出现全训练未见的城镇、街道、户型或设计类别；单独 BLOCK 有新字符串，但应按完整楼栋键解释。城镇/户型分布的边际差异较小并不证明条件分布或未来租金稳定。

![数值分布比较](../proposal/figures/eda_numeric_shift.png)

数值差异使用 SMD = (test 均值 − train 均值) / sqrt((train 样本方差 + test 样本方差)/2)：

{table(nums.rename(columns={'value':'SMD','test_outside_train_range':'test 超出 train 范围行数'}),4)}

与最近训练月份相比，面积 SMD 为 {float(drift[(drift.field=='FLOOR_AREA_SQM')&(drift.reference!='all_train')&(drift.metric=='standardized_mean_difference')].iloc[0].value):.4f}，楼龄 SMD 为 {float(drift[(drift.field=='AGE_YEARS')&(drift.reference!='all_train')&(drift.metric=='standardized_mean_difference')].iloc[0].value):.4f}。近期窗口面积最大值较小，测试有 {int(drift[(drift.field=='FLOOR_AREA_SQM')&(drift.reference!='all_train')&(drift.metric=='standardized_mean_difference')].iloc[0].test_outside_train_range)} 行超出该近期面积范围；这不等同超出全训练范围。完整均值、范围、未见类别数量与类别占比差见 [feature_drift.csv](../experiments/eda/core/feature_drift.csv) 和 [category_distribution_comparison.csv](../experiments/eda/core/category_distribution_comparison.csv)。

**推测**：时间趋势及较长跨度可能比单一房屋字段的边际漂移更影响泛化；部分年龄均值变化是样本批准年份后移带来的机械变化。**待验证**：租金的条件关系是否改变、趋势是否延续、是否存在可复现季节性、是否应采用近期窗口或时间权重。测试无标签，本次无法观测测试目标分布，也无法验证这些推测。

团队指南建议的 2021-01–2024-09 / 2024-10–2025-03 窗口在当前数据下有 {sw['fit_rows']:,} / {sw['holdout_rows']:,} 行，后段有 {sw['holdout_new_block_rows']} 条记录的楼栋未在前段出现。本报告只核对可执行样本量，没有冻结协议或运行验证。B/C 仍须共同确认滚动窗口与接近实际最大间距的长期压力窗口，统一每折可用信息时间；不能只用随机划分或短期分数代表长跨度效果。

## 5. 重复记录、重复交易与楼栋覆盖

{table(ds)}

`all_available_fields` 在 train 包含目标，但 test 不包含目标，两者不是同一重复判定信息量；`X_only` 在两集都排除目标。`repeated_groups` 是至少两行的组数，`rows_in_repeated_groups` 是这些组中全部行数，`excess_rows` 才是超出每组首行的数量。block_month 指规范化前/后的城镇、楼号、街道和批准月。完整表还包含不分月份的楼栋重复统计：[duplicates.csv](../experiments/eda/core/duplicates.csv)。

规范化后，train 中相同全部 X 却有不同租金的组有 {s['identical_X_multi_target_groups']:,} 个，涉及 {s['identical_X_multi_target_rows']:,} 行。它们可能对应不同单元、不同合同或其他未观测条件，不能把这些行自动平均、只保留首条或认定为目标错误。相同楼栋在不同月份重复出现更不能按地址删掉。无单元编号、交易 ID 和日级日期时，不能证实整行相同就是源数据重复。

两集规范化楼栋覆盖：

{table(pd.DataFrame(s['block_coverage']))}

全训练 {cv['train_unique_blocks']:,} 栋、测试 {cv['test_unique_blocks']:,} 栋，共享 {cv['shared_unique_blocks']:,} 栋；测试已见楼栋行数 {cv['test_seen_rows']:,}/{s['rows']['test']:,}（{cv['test_seen_rows']/s['rows']['test']:.3%}），新楼栋 {cv['test_new_blocks']} 栋、{cv['test_new_rows']} 行（{cv['test_new_rows']/s['rows']['test']:.3%}）。若只保留最近训练窗口，测试有 {recent_cv['test_new_rows']:,} 行找不到该窗口内的楼栋历史；这说明近期训练可能在时效与覆盖之间存在权衡，仍需实验。

2025-03 的 test 有 {s['test_rows_identical_X_in_train']} 行全部 X 在 train 出现，见 [cross_split_identical_X_test_rows.csv](../experiments/eda/core/cross_split_identical_X_test_rows.csv)。没有测试目标，不能证明目标泄漏，也不应当作可直接复制租金的答案。未来的楼栋目标统计必须严格来自当时可见训练历史，不能利用验证目标、同月不明先后的目标或当前记录自身。每折都应重新计算已见/未见楼栋标签，而非沿用全训练的覆盖率。

'''
    aux=(output/'auxiliary'/'report_section.md').read_text()
    # Make audit file references clickable relative to docs/, preserving the fragment as a standalone source.
    text+=aux.replace('## 辅助数据审计（实际 CSV；仅用于质量审计）','## 6. 辅助数据连接、质量与时间边界')
    text+='''

完整辅助字段/范围和类别频数分别见 [field_overview.csv](../experiments/eda/auxiliary/field_overview.csv)、[categorical_frequencies.csv](../experiments/eda/auxiliary/categorical_frequencies.csv)。连接率、未匹配及冲突见 [hdb_match_rates.csv](../experiments/eda/auxiliary/hdb_match_rates.csv)、[hdb_unmatched_or_ambiguous.csv](../experiments/eda/auxiliary/hdb_unmatched_or_ambiguous.csv)、[hdb_key_conflicts.csv](../experiments/eda/auxiliary/hdb_key_conflicts.csv)。匹配率是地址匹配成功率，不是特征可用率或预测正确率。

## 7. 建议清洗规则（待团队审阅；原始文件保持不变）

| 环节 | 建议派生层规则 | 不确定性及后续校验 |
| --- | --- | --- |
| 月份、数值 | 日期严格解析；面积/租金检查有限正值；保留原始行序和来源行号 | 异常先标记；测试无目标字段，不能误当缺失目标填补 |
| 文本、户型 | 使用上述确定性规则；楼号及邮编保留字符串；原值→规范值留表 | 不凭猜测合并语义不同类别；折内学词表，未见类别走明确回退 |
| 常量 | 派生建模输入可去掉 FURNISHED 和 FEE，保留原始字段与常量审计 | 后续版本若不再常量需重新审阅 |
| 负年龄 | 保留原年份、计算年龄与异常指示；核实后再确定缺失/修正规则 | 禁止直接改成零或从未来信息修正；本次保留该行 |
| 尾部与重复 | 默认保留；IQR/同 X 只标记；验证集保留完整分布 | 如果团队提出去重、权重或截尾，只能在统一时间窗口下做对照，并说明删除依据 |
| HDB 连接 | 规范化三字段键，validate='many_to_one'，检查输出行数和顺序 | 当前无冲突；后续有一对多或未匹配须显式报告，不能静默选首条 |
| HDB 缺失/年代 | NIL 邮编与 -1 楼层/竣工年转为派生缺失并保留指示 | 不混用竣工年与租约年，不把 MAX_FLOOR 当成出租单元实际楼层 |
| 设施 | 先定义站点/学校的实体计数、学段和生效时间，再计算距离 | 若无可靠历史日期，严格时点实验暂不使用“当时已开放”特征；不得用当前 open 回填历史 |
| 宏观 | 先确认固定起点或逐月滚动语境，按实际公布时间限制输入 | 不用未来实际值、未来复权/公司筛选结果、反向填补或全期间股票筛选；仅滞后一个月不自动安全 |

## 8. 值得后续测试的特征或方法假设

以下是证据驱动的开放清单，既非有效方案结论，也非固定候选数量。所有方案必须在相同时间窗口、相同完整验证样本和 SGD 原尺度 RMSE 下比较，必要时补充 MAE 与分组误差。

| EDA 依据 | 后续可检验假设 | 检验与失败判据 |
| --- | --- | --- |
| 训练月均明显变化，而规范化后户型/城镇边际构成差异较小；同户型也随月变化 | 月序趋势、分段时间效应、时间×户型/城镇交互，或近期窗口/样本权重可能有用 | 比较多个向前窗口及长跨度窗口；若只短期有效或新楼栋覆盖明显受损，不宣称普遍收益 |
| 面积和年龄的边际关系弱到中等、分箱非严格线性 | 面积/年龄的非线性表达与户型、设计类别交互值得测试 | 先与同输入的简单表达作受控比较；不能用全数据图的拐点反复针对验证调参 |
| 楼栋覆盖很高但同 X 仍有不同租金 | 楼栋/城镇分层信息、带平滑的历史统计可能提供条件信息 | 严格按时间或折外构造，排除自身和未可知同月目标；同时报告已见、未见楼栋及稀疏组表现 |
| HDB 保守连接全覆盖，含坐标、楼层和规划分区 | 地理位置、规划区、楼栋属性可能补充仅城镇层面的信息 | 按组增加并消融；先处理占位缺失和历史可用性，MAX_FLOOR 仅代表楼栋上限 |
| 设施提供坐标但状态和日期不完整 | 经时点核实的交通/学校/商场距离或数量可能提供信息 | 先补齐生效日期；统一距离单位、学段筛选及换乘站计数；缺少日期时不能宣布是无泄漏特征 |
| 经济表覆盖长，但时间和价格一致性存在问题 | 在预测起点已知的滞后经济指标可能解释部分时间变化 | 先审来源和公布日，再与仅时间变量对照；固定起点不能偷用起点后的实际宏观值 |
| 租金有尾部和离散集中、同 X 不完全确定目标 | 原租金与变换目标、不同尾部处理或鲁棒拟合值得受控比较 | SGD 原尺度 RMSE 和尾部误差为共同依据；检验反变换偏差，保留验证尾部，未验证前不推荐某一损失/变换 |
| 罕见户型/设计类别样本少 | 合理共享统计、平滑或未见类别回退可能比极细分组稳定 | 词表、频数阈值和统计只从训练折学；记录罕见组样本量，不能仅看总体 RMSE |

此阶段没有任何 RMSE 成绩，也没有方法排名。季节性、设施收益、宏观收益和稳健损失收益均仍是待验证问题。

## 9. 未解决问题与交付索引

- 2025-03 同时出现在两集、说明截至 2026-06 但测试到 2026-07：确认切分规则与课程说明差异，不能推断月内先后。
- 预测语境是一次固定起点预测整段未来，还是每月更新已知外部信息：这决定经济数据可用边界，须先由团队统一。
- 负年龄、HDB 竣工年晚于租约年、NIL/-1、COE 的类别缺口与零值、股票缺失/OHLC 冲突的来源含义仍待核实。
- 设施历史版本、开业/开放/迁址日期和股票 as-of 复权/公司名单来源缺失；本次仅审计快照，不宣称可历史回放。
- 无单元/交易标识，不能确认完全相同行或同月同栋多行的真实重复语义；去重决策需证据，不能仅为分数作删行。
- 统一时间验证窗口、长跨度压力测试和清洗版本仍需团队冻结；本报告不代替 B/C 的协议确认。

交付：入口 [run_eda.py](../notebooks/run_eda.py)；核心、辅助、图表与报告生成模块均位于 `notebooks/`；[依赖](../notebooks/requirements-eda.txt)；[运行说明](../notebooks/README_EDA.md)；[运行清单](../experiments/eda/run_manifest.json)。核心明细位于 `experiments/eda/core/`，辅助明细位于 `experiments/eda/auxiliary/`。上述图表位于 `proposal/figures/eda_*.png`，所有图可重新生成。

## 10. 给 B/C 的简短交接清单

1. 先复用确定性文本/户型规范化与三字段楼栋键；保留原始行序、原值及异常指示。常量可移出输入，尾部和重复默认保留，负年龄先标记待核实。
2. HDB 连接强制 many-to-one，检查行数/顺序；NIL/-1 设派生缺失。设施生效日期未核实前，不能使用“当时已开放”假设；宏观输入严格服从预测起点与公布时间。
3. 与团队冻结共同的向前时间窗口、滚动窗口和长跨度压力测试。所有填补、词表、缩放、异常阈值及目标统计仅在相应训练折拟合。
4. 在同一完整验证集和原尺度 RMSE 下检验时间、房屋交互、历史楼栋、地理/设施、经济及目标处理假设，逐组消融并保留失败记录；没有默认有效方案，也不限定候选数。
5. 同时交付按月份/预测间距、城镇、户型、租金区间、已见/未见楼栋的误差和样本量，说明近期窗口的覆盖损失与长跨度结果是否一致。

阶段一到此停止，等待团队审阅。后续模型实验、Kaggle、proposal 幻灯片及最终报告均未开展。
'''
    (root/'docs'/'EDA_REPORT.md').write_text(text)
