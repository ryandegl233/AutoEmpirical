# 外部证据读取工具

实现位置：`Benchmark/src/external_evidence_tools.py`。独立于分类和预注册冻结证据图；不会调用LLM或自动把抓到的材料标成可信原因。

## 使用

请求文件是JSON数组，每项只有`request_id`、`case_id`、`request`。例如：

```json
[
  {
    "request_id": "case7-err-log",
    "case_id": "7",
    "request": {
      "tool": "read_page",
      "url": "https://github.com/tensorflow/tfjs/files/7369103/err.log"
    }
  }
]
```

在仓库根目录运行（`python`指安装了Pillow的Python）：

```powershell
python Benchmark/scripts/collect_external_evidence.py --requests requests.json --output-dir Benchmark/cache/capture --results Benchmark/cache/capture/summary.json
python Benchmark/scripts/collect_external_evidence.py --requests requests.json --output-dir Benchmark/cache/capture --results Benchmark/cache/capture/replay.json --offline
python -m unittest discover -s tests -p test_external_evidence_tools.py -v
```

同一个输出目录保存一个冻结快照，已缓存URL不重新抓取。需要新抓取时间时换输出目录。结果路径不得覆盖旧文件。返回码：0全部成功，1部分或全部读取不成功（详见结果），2参数／文件错误。

## 四项接口

| tool | 参数 |
|---|---|
| `read_page` | `url`；可选`cursor`（提取后字符偏移）、`member`（ZIP内文件名） |
| `read_issue` | GitHub issue或PR的`url`，自动读取最多3页评论；PR另读审查评论 |
| `read_code` | `repo`、`ref`、`path`，可选`start_line`、`end_line`；或PR／commit的`url` |
| `read_image` | `url`；支持单帧PNG、JPEG、WebP，保存原图 |

`read_code`的ref先解析为commit SHA，再读取固定源码。PR取得base/head SHA后，使用这两个固定提交的compare接口取差异，另记录真正的diff起点merge-base，避免PR更新后版本和差异错配。API未给patch、文件数不符或达到compare的300文件上限时标partial；patch内容完整性没有做独立验证。固定提交也不自动代表实验允许的历史时点。[GitHub compare说明](https://docs.github.com/en/rest/commits/commits#compare-two-commits)

输出目录包含：`raw/`原始HTTP响应、`responses/`来源及hash索引、`assets/`原图、`results/`单次工具结果。所有项都有来源、抓取时间和hash；评论另有作者与发布时间。`temporal_status=unverified`表示尚未核对实验历史截止时间。

## 图片怎么给模型

```python
from Benchmark.src.external_evidence_tools import EvidenceTools

tools = EvidenceTools('Benchmark/cache/capture', offline=True)
result = tools.run({'tool': 'read_image', 'url': image_url})
image = result['items'][0]
gemini_part = tools.image_part(image, 'gemini')
openai_part = tools.image_part(image, 'openai')
```

`gemini_part`包含`inlineData`；`openai_part`包含base64 data URL。把它与文字一起放进对应接口的多模态content列表。这里构造实际图像输入，不做OCR，不请求模型。实际模型／网关是否接受需另测；不能把成功构造消息说成已验证模型看图。

Gemini图片输入格式见[官方文档](https://ai.google.dev/gemini-api/docs/image-understanding)；固定版本源码的ref机制见[GitHub文档](https://docs.github.com/en/rest/repos/contents)。

## 限制

每次HTTP读取上限10MB，单次请求约20秒并限制跳转；网页返回最多24000字符并给续读位置，源码超长要求缩小行范围。图片上限2500万像素，ZIP最多1000项、单文件10MB，只在内存读取不解压到磁盘。CLI每例每清单最多8次工具调用；这不是已接入framework的跨轮次预算控制器。

公开HTTP(S)只读访问，拒绝本地地址，可用`--allow-host`列出允许域名（含跳转目标）。生产部署仍应配合网络出口策略，不把应用层地址检查当成完整的网络隔离。

GitHub可通过`GITHUB_TOKEN`或`GH_TOKEN`提供访问令牌，只发给HTTPS api.github.com；不读取其他软件的凭据。403、429、404、超时、登录页和格式不支持各自记录；公开API限流时不绕过。

搜索、浏览器动态渲染、PDF/其他压缩格式、图片裁剪和模型自主调度未在本轮实现。下载成功只说明材料可读取；它是否与本例、版本和时间对应，需要后续核查。
