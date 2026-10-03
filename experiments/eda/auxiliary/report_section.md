## 辅助数据审计（实际 CSV；仅用于质量审计）

统计来源：`experiments/eda/auxiliary/summary.json`、`field_overview.csv` 和各专项表。全量辅助文件用于检查覆盖边界，未据此创建预测特征、填补训练数据或使用测试目标。原始文件 SHA-256 在运行前后核对一致。

| 文件 | 行数 | 列数 | 空值单元格 | 整行重复多余行 | 时间覆盖 |
| --- | --- | --- | --- | --- | --- |
| sg-coe-prices.csv | 539 | 8 | 0 | 0 | 2021-01–2026-08，68 个月 |
| sg-hdb-block.csv | 9464 | 11 | 0 | 0 | 快照；无生效/采集日期 |
| sg-mrt-stations.csv | 243 | 8 | 0 | 0 | 快照；无生效/采集日期 |
| sg-schools.csv | 337 | 14 | 0 | 0 | 快照；无生效/采集日期 |
| sg-shopping-malls.csv | 113 | 3 | 0 | 0 | 快照；无生效/采集日期 |
| sg-stock-prices.csv | 91642 | 9 | 510 | 0 | 2021-01-04–2026-07-30，67 个月 |

字段明细（逐列推断类型、范围、缺失、唯一值、零/负值和常量标记在 `field_overview.csv`；字符串范围按字典序，完整频数在 `categorical_frequencies.csv`）：

- `sg-coe-prices.csv`：`YEAR`, `CATEGORY`, `MONTH`, `ROUND`, `QUOTA_PREMIUM`, `PREVAILING_QUOTA_PREMIUM`, `QUOTA`, `BIDS_RECEIVED`。
- `sg-hdb-block.csv`：`TOWN`, `BLOCK`, `STREET`, `POSTAL_CODE`, `LATITUDE`, `LONGITUDE`, `MAX_FLOOR`, `YEAR_COMPLETED`, `SUBZONE`, `PLANNING_AREA`, `REGION`。
- `sg-mrt-stations.csv`：`CODE`, `NAME`, `STATUS`, `LATITUDE`, `LONGITUDE`, `SUBZONE`, `PLANNING_AREA`, `REGION`。
- `sg-schools.csv`：`NAME`, `URL`, `STREET`, `POSTAL_CODE`, `MRT_STATIONS`, `BUS_LINES`, `PLANNING_AREA`, `REGION`, `TYPE_CODE`, `NATRUE_CODE`, `SESSION_CODE`, `MAINLEVEL_CODE`, `LATITUDE`, `LONGITUDE`。
- `sg-shopping-malls.csv`：`NAME`, `LATITUDE`, `LONGITUDE`。
- `sg-stock-prices.csv`：`NAME`, `SYMBOL`, `DATE`, `OPEN`, `HIGH`, `LOW`, `CLOSE`, `ADJUSTED_CLOSE`, `VOLUME`。

### HDB 楼栋连接与占位缺失

仅对内存中的 `TOWN`、`BLOCK`、`STREET` 做 Unicode NFKC、首尾空白去除、小写化、连续空白压为一个空格。楼号保留字母后缀；不转数值，不猜街道别名，不模糊连接。以 `TOWN + BLOCK + STREET` 为主键，另外审计 `BLOCK + STREET` 的 fallback 可能性。

| split | mode | matched_rows | rows | row_match_pct | matched_unique_keys | unique_keys | unique_key_match_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | raw_town_block_street | 119219 | 150000 | 79.4793% | 9344 | 17413 | 53.6611% |
| test | raw_town_block_street | 39518 | 50000 | 79.0360% | 8674 | 14314 | 60.5980% |
| train | normalized_town_block_street | 150000 | 150000 | 100.0000% | 9406 | 9406 | 100.0000% |
| test | normalized_town_block_street | 50000 | 50000 | 100.0000% | 8940 | 8940 | 100.0000% |
| train | normalized_block_street | 150000 | 150000 | 100.0000% | 9406 | 9406 | 100.0000% |
| test | normalized_block_street | 50000 | 50000 | 100.0000% | 8940 | 8940 | 100.0000% |

原始模式的“唯一键”分母保留大小写写法差异；规范化模式分母为规范化楼栋数，两者不能直接当作同一楼栋分母比较。两套规范化键均达到全量匹配，因此没有必要启用 fallback。三种模式在 HDB 表的重复键、一对多键及冲突行均为 0；规范化后未匹配和歧义楼栋均为 0。原始失配全部经格式规范化恢复，未观察到需要街道改名或城镇猜测的证据。

`hdb_key_crosswalk.csv` 保留原始键、规范化键、输入 split、交易行数和匹配 HDB 零基行号；`hdb_normalized_inventory.csv` 记录每栋在 train/test 的使用行数。`hdb_key_conflicts.csv`、`hdb_unmatched_or_ambiguous.csv` 保留表头，即使目前为空也作为下次运行的审计接口。后续生产连接必须使用 `validate='many_to_one'`，遇到冲突停止并保留全部候选，不可静默选首条。

HDB 表没有 pandas 空值，但 `POSTAL_CODE='NIL'` 有 285 行，`MAX_FLOOR=-1` 和 `YEAR_COMPLETED=-1` 各 11 行。有效竣工年份为 1949–2021，有效最大楼层数为 2–50。这些哨兵值应在派生数据中转缺失并保留缺失指示；邮编保持字符串。匹配成功不代表每个属性都有有效值。

| split | rows_joined | rows_postal_nil | rows_floor_minus_one | rows_completion_minus_one | rows_completion_after_approval_year | rows_completion_after_lease_year | buildings_completion_after_lease_year | max_completion_minus_lease_years |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| train | 150000 | 5259 | 159 | 159 | 0 | 422 | 26 | 10 |
| test | 50000 | 1739 | 0 | 0 | 0 | 152 | 25 | 10 |

上述竣工年晚于起租契年份的记录已单列 `hdb_train_completion_after_lease.csv` / `hdb_test_completion_after_lease.csv`，属于跨表口径核查项，不能未经查证就用一个年份覆盖另一个。`YEAR_COMPLETED` 是竣工年份，不能视为该辅助记录的生效时间。文件缺少历史版本、拆除和地址变更日期；地理位置、楼层等是否可追溯到每个批准月仍需确认。

### MRT、学校、商场：快照不是历史可用性证明

- **MRT**：243 条线路/站代码记录，`STATUS` 为 open 231、planned 12；只有 192 个站名和 192 个坐标，按坐标重复的多余记录 51 条；11 个站名同时含 open 与 planned。多代码可能代表换乘线路，不是整行重复，不能按记录数直接计“站点数”。没有开放/规划公布日期，当前 open 不代表 2021 年已开放。历史“最近已开放站”或线路数量必须先取得可靠生效日期；否则只能作为明确注明静态快照假设的待验证特征，不得声称严格时点有效。planned 不应混入已开放距离。
- **学校**：337 所，`MAINLEVEL_CODE='primary'` 仅 179 所，文件还包含中学、初院和混合学段，不能将全表称为小学。333 个坐标，坐标重复多余 4 条可能为共址，需按目标定义决定计数口径。可审议距离、学段及学校类型；开校、迁址、合并日期缺失。`NATRUE_CODE` 是原文件列名拼写，派生层若改名需明确映射；`REGION` 的学校分区（east/north/south/west）与 HDB/MRT 地理分区词表不一致，不能直接互换或连接。
- **商场**：113 条、113 个名称，整行/坐标均无重复；只有名称与经纬度。可审议距离或邻域数量，无法单凭此表判定历史月份的开业、关业与改造状态。

三表的经纬度均通过全球合法范围检查（纬度 ±90、经度 ±180）；这只排除范围错误，不能证明地理定位正确。设施距离计算须使用球面距离并标注米/千米，且明确站点去重和生效日期口径。

### COE 与股票：时间覆盖和泄漏边界

**COE 事实**：539 行，覆盖 2021-01–2026-08 共 68 个月，类别为 a, b, c, e（原文件无 d）。按“月×类别×轮次”无重复；若在全局月份内假定每类别每月两轮，缺 5 个格子，清单见 `coe_missing_month_category_rounds.csv`，不能据此断言数据错误（末月可能不完整）。美元符号和千分位逗号去掉后，四个数值列解析失败均为 0；解析后的范围见 `coe_numeric_parse_audit.csv`。`PREVAILING_QUOTA_PREMIUM` 的零值按类别为 {"e": 134}，零不能未经确认直接当作经济意义上的零或统一填补。

**股票事实**：91642 行、77 个 symbol，2021-01-04–2026-07-30 共 67 个月，`SYMBOL + DATE` 无重复。每月 symbol 数为 53–77；53 个覆盖全部 67 个月，7 个首条日期晚于训练结束月。价格各列和 VOLUME 各缺 85 行（相同行），日期分布为 {"2026-05-01": 43, "2026-06-01": 42}；不能把周末/节假日的自然无交易记录等同缺失，逐 symbol 日历还需结合交易所。VOLUME=0 有 2623 行，应与未知区分。HIGH 低于 OPEN/CLOSE/LOW 的行数为 46，LOW 高于 OPEN/CLOSE/HIGH 的行数为 26，合计 72 个不同记录，见 `stock_ohlc_inconsistencies.csv`；原因待核查，未自动修复。

经济表覆盖与核心数据的关系：

| table | rows_after_train_end_month | rows_after_test_end_month | train_missing_months | test_missing_months |
| --- | --- | --- | --- | --- |
| coe | 131 | 3 | [] | [] |
| stock | 25126 | 0 | [] | [] |

**风险与待决策**：租赁只有批准月份，没有日。COE 两轮结果、当月股票收盘或月均不保证在该月某笔批准前已知，不能直接把当月汇总批准为无泄漏输入。若团队采用“每个月开始时滚动预测”，可测试仅用此前已经公布的滞后指标；还需核实发布时间，严禁反向填补。若任务定义为“2025-03 一次性预测至 2026-07”，则 2025-04 起实际观测的经济指标在该起点全部属于未来，简单滞后一个月也不够，须在每个验证截止点冻结可用历史。COE 2026-08 记录甚至晚于最后测试月，不得用于任何测试月特征。

股票 `ADJUSTED_CLOSE` 的历史复权值可能包含后续分红/拆股调整；其他价格是否做过拆股处理、公司名单是否按未来市值筛选，以及退市/新股带来的存活偏差，均不能由当前 CSV 证明或排除。严格时点回放应核对数据来源和 as-of 历史版本；当前阶段只记录风险，不从全期间选择“最佳股票”，不利用验证后走势选择特征。不同交易所、币种、上市日期及缺失模式也可能使简单等权平均缺少一致含义。

### 交给 B/C 的辅助数据事项

先使用本脚本保守主键规则和 many-to-one 校验，保留原始 CSV；将 HDB 哨兵值显式标记缺失。确认设施历史生效日期与换乘站计数口径，确认经济指标在长跨度固定起点和逐月滚动两种预测语境中的可用边界。楼栋/位置、历史可用设施、滞后经济指标都只是待测试假设，应在团队统一时间切分和 RMSE 下逐组消融；尚无任何预测效果结论。
