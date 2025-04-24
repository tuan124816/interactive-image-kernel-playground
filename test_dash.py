import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
from plotly.graph_objs import layout, Scatter
import numpy as np
import cv2
from PIL import Image
import io
import base64

app = dash.Dash(__name__)
app.title = "Hover Convolution Explorer"

default_kernel = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
]) / 16

# Layout
app.layout = html.Div([
    html.H2("Hover-Based Convolution Explorer"),
    html.Div([
        dcc.Upload(id="upload", children=html.Button("Upload Image"), accept="image/*"),
        html.Div(id="upload-status"),
        dcc.Store(id="stored-array"),
    ]),
    html.Div([
        html.H4("Adjust Kernel (3x3)"),
        html.Div([
            html.Div([
                dcc.Input(id=f"k{i}{j}", type="number", step=0.01,
                          value=float(default_kernel[i, j]), style={"width": "80px"})
                for j in range(3)
            ], style={"display": "flex", "gap": "10px", "marginBottom": "5px"})
            for i in range(3)
        ])
    ]),
    html.Div([
        html.Div([
            dcc.Graph(id="input-img", config={"staticPlot": True}),
            html.H5("Input Image")
        ], style={"display": "inline-block", "width": "48%"}),
        html.Div([
            dcc.Graph(id="output-img", config={"staticPlot": True}),
            html.H5("Output Image (Filtered)")
        ], style={"display": "inline-block", "width": "48%"}),
    ]),
    html.Pre(id="math", style={"whiteSpace": "pre-wrap", "fontSize": "16px", "marginTop": "20px"})
])


@app.callback(
    Output("upload-status", "children"),
    Output("stored-array", "data"),
    Input("upload", "contents"),
    prevent_initial_call=True
)
def handle_upload(contents):
    if contents is None:
        return "", None
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = Image.open(io.BytesIO(decoded)).convert("L")
    return "Image uploaded!", np.array(img).tolist()


@app.callback(
    Output("input-img", "figure"),
    Output("output-img", "figure"),
    Input("stored-array", "data"),
    [Input(f"k{i}{j}", "value") for i in range(3) for j in range(3)],
)
def update_images(img_array, *kernel_vals):
    if img_array is None:
        return dash.no_update, dash.no_update

    img_array = np.array(img_array)
    kernel = np.array(kernel_vals).reshape((3, 3))

    # Apply convolution
    img_array = np.array(img_array).astype(np.float32)
    filtered = cv2.filter2D(img_array, -1, kernel)

    fig_in = px.imshow(img_array, binary_string=True)
    fig_in.update_layout(dragmode=False, margin=dict(l=0, r=0, t=0, b=0))

    fig_out = px.imshow(filtered, binary_string=True)
    fig_out.update_layout(dragmode=False, margin=dict(l=0, r=0, t=0, b=0))

    return fig_in, fig_out


@app.callback(
    Output("math", "children"),
    Input("input-img", "hoverData"),
    [State(f"k{i}{j}", "value") for i in range(3) for j in range(3)],
    State("stored-array", "data")
)
def update_math(hoverData, *args):
    kernel_vals = args[:9]
    array = args[9]
    if array is None or hoverData is None:
        return "Hover over the image to see convolution math"
    img_array = np.array(array)
    kernel = np.array(kernel_vals).reshape((3, 3))
    point = hoverData["points"][0]
    x, y = point["x"], point["y"]

    if x < 1 or y < 1 or x > img_array.shape[1] - 2 or y > img_array.shape[0] - 2:
        return "Hover near the center for convolution preview"

    region = img_array[y-1:y+2, x-1:x+2]
    terms = [f"{region[i,j]}×{round(kernel[i,j],3)}" for i in range(3) for j in range(3)]
    result = round(np.sum(region * kernel), 2)
    equation = " + ".join(terms)
    return f"Convolution at ({x},{y}):\n{equation} = {result}"


@app.callback(
    Output("input-img", "figure", allow_duplicate=True),
    Output("output-img", "figure", allow_duplicate=True),
    Input("input-img", "hoverData"),
    State("stored-array", "data"),
    [State(f"k{i}{j}", "value") for i in range(3) for j in range(3)],
    prevent_initial_call=True
)
def highlight_hover(hoverData, img_array, *kernel_vals):
    if hoverData is None or img_array is None:
        raise dash.exceptions.PreventUpdate

    x = hoverData["points"][0]["x"]
    y = hoverData["points"][0]["y"]

    img_array = np.array(img_array).astype(np.float32)
    kernel = np.array(kernel_vals).reshape((3, 3))
    filtered = cv2.filter2D(img_array, -1, kernel)

    fig_in = px.imshow(img_array, binary_string=True)
    fig_out = px.imshow(filtered, binary_string=True)

    if 1 <= x < img_array.shape[1] - 1 and 1 <= y < img_array.shape[0] - 1:
        # Draw 3x3 region on input
        for dx in range(-1, 2):
            for dy in range(-1, 2):
                fig_in.add_shape(
                    type="rect",
                    x0=x + dx - 0.5,
                    y0=y + dy - 0.5,
                    x1=x + dx + 0.5,
                    y1=y + dy + 0.5,
                    line=dict(color="red", width=1),
                )

        # Draw output pixel
        fig_out.add_shape(
            type="rect",
            x0=x - 0.5,
            y0=y - 0.5,
            x1=x + 0.5,
            y1=y + 0.5,
            line=dict(color="red", width=2),
        )

    fig_in.update_layout(dragmode=False, margin=dict(l=0, r=0, t=0, b=0))
    fig_out.update_layout(dragmode=False, margin=dict(l=0, r=0, t=0, b=0))

    return fig_in, fig_out

if __name__ == "__main__":
    app.run(debug=True)