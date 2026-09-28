import re
import json

custom_css = """
/* 隐藏页脚 */
footer { display: none !important; }
/* 增大字体 */
.prose { font-size: 18px !important; line-height: 1.6 !important; }
textarea { font-size: 16px !important; }
/* 强行撑满全屏 */
.gradio-container { max-width: 100% !important; margin: 0 !important; padding: 1% 2% !important; }

/* 核心对齐逻辑：让两边的上部大窗口强行等高，下部小窗口天然对齐 */
.main-window {
    height: 80vh !important;
    overflow-y: auto !important;
}
"""

def fetch_file_tree():
    import sys, os
    if "/home/ubuntu/tencent-docs-mcp" not in sys.path:
        sys.path.append("/home/ubuntu/tencent-docs-mcp")
    try:
        from client import TencentDocsClient
        client = TencentDocsClient(
            os.getenv("TENCENT_DOCS_CLIENT_ID", ""),
            os.getenv("TENCENT_DOCS_CLIENT_SECRET", ""),
            "https://docs.qq.com"
        )
        data = client.list_docs(limit=50)
        items = data.get("list", [])
        
        container_style = (
            "padding: 20px; "
            "background-color: var(--background-fill-secondary); "
            "color: var(--body-text-color); "
            "border: 1px solid var(--border-color-primary); "
            "border-radius: var(--radius-lg); "
            "font-size: 16px; "
            "height: 100%; "
            "box-sizing: border-box;"
        )
        
        if not items:
            return f"<div style='{container_style}'><h3 style='margin-top:0; font-size:18px;'>🗂️ 文档结构树</h3>暂无文档或未授权。</div>"
            
        html = f"<div style='{container_style}'><h3 style='margin-top:0; font-size:18px;'>🗂️ 文档结构树</h3><ul style='list-style-type: none; padding-left: 0;'>"
        html += "<li style='margin-bottom:15px; font-size: 17px;'>📂 <b>我的腾讯文档</b></li>"
        
        folders = [item for item in items if item.get('type') == 'folder']
        files = [item for item in items if item.get('type') != 'folder']
        
        link_style = "text-decoration:none; color: var(--body-text-color);"
        
        for f in folders:
            html += f"<li style='margin-left:15px; margin-bottom:12px;'>📁 <a href='{f['url']}' target='_blank' style='{link_style}'><b>{f['title']}</b></a></li>"
        for f in files:
            icon = "📊" if f.get('type') == 'sheet' else "📄"
            html += f"<li style='margin-left:15px; margin-bottom:12px;'>{icon} <a href='{f['url']}' target='_blank' style='{link_style}'>{f['title']}</a></li>"
            
        html += "</ul></div>"
        return html
    except Exception as e:
        return f"<div style='padding:10px;color:red;'>获取文档树失败: {str(e)}</div>"

def extract_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)
    return str(content) if content is not None else ""

def build_ui(chat_fn):
    import gradio as gr
    import asyncio
    
    def bot_response(history):
        history = history or []
        if not history:
            yield history
            return
        
        last_item = history[-1]
        raw_msg = last_item.get("content", "") if isinstance(last_item, dict) else getattr(last_item, "content", "")
        user_message = extract_text(raw_msg)
        hist_api = history[:-1]
        
        # 立即追加助手占位状态，避免界面空转假死
        history.append({"role": "assistant", "content": "🤔 正在分析处理，正在调用相关文档接口..."})
        yield history
        
        try:
            response = chat_fn(user_message, hist_api)
        except Exception as e:
            response = f"发生错误: {e}"
            
        history[-1]["content"] = response
        yield history


    def user_input_handler(user_message, history):
        history = history or []
        history.append({"role": "user", "content": user_message})
        return "", history

    with gr.Blocks(title="腾讯文档智能助手", fill_height=True) as demo:
        with gr.Row():
            # ================= 左侧 =================
            with gr.Column(scale=1, min_width=300):
                # 上半部分：文件树 (绝对等高)
                file_tree = gr.HTML(
                    value="<div style='padding:20px; font-size:16px; height: 100%; box-sizing: border-box;'>加载中...</div>", 
                    elem_classes=["main-window"]
                )
                # 下半部分：刷新按钮
                refresh_btn = gr.Button("🔄 刷新文档树", size="lg")
                
            # ================= 右侧 =================
            with gr.Column(scale=4):
                # 上半部分：聊天框 (绝对等高)
                chatbot = gr.Chatbot(
                    elem_classes=["main-window"],
                    show_label=False,
                    placeholder="<div style='text-align: center; color: gray; margin-top: 20vh; font-size: 16px; line-height: 1.8;'>💡 <b>专属长期记忆提示</b><br><br>您可以对我说「记住，以后我说 A 就代表文档 B」来训练我的长期记忆。<br><br><span style='color: #a0a0a0; font-size: 15px;'><i>例如：【 记住，以后我说“封边条”就是指“封边条250424” 】</i></span><br><br>系统将为您自动记录规则，越用越懂您！</div>"
                )
                # 下半部分：输入框
                with gr.Row():
                    txt = gr.Textbox(
                        show_label=False,
                        placeholder="在此输入您的问题... (按 Enter 发送，Shift+Enter 换行)",
                        container=False,
                        lines=1,
                        max_lines=5,
                        scale=7
                    )
                    submit_btn = gr.Button("发送", variant="primary", scale=1)
        
        # 事件绑定
        demo.load(fetch_file_tree, inputs=None, outputs=file_tree)
        refresh_btn.click(fetch_file_tree, inputs=None, outputs=file_tree)
        
        # 聊天事件绑定 (支持回车和点击发送，设置并发度为 2 防止内存超限)
        txt.submit(user_input_handler, [txt, chatbot], [txt, chatbot]).then(
            bot_response, chatbot, chatbot, concurrency_limit=2
        )
        submit_btn.click(user_input_handler, [txt, chatbot], [txt, chatbot]).then(
            bot_response, chatbot, chatbot, concurrency_limit=2
        )
        
    return demo
