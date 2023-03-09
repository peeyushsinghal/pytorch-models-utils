from torchinfo import summary


def print_summary(model, input_size, device ='cpu',  batch_size=8):
    model = model.to(device=device)
    s = summary(
        model,
        input_size=(batch_size, *input_size),
        verbose=0,
        col_names=[
            "num_params",
            "kernel_size",
            "input_size",
            "output_size"
        ],
        row_settings=["var_names"],
    )

    print(s)
    return