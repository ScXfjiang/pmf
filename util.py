def show_elapsed_time(start, end):
    sec = int(end - start)
    hour = sec // 3600
    sec = sec - hour * 3600
    min = sec // 60
    sec = sec - min * 60
    print(
        "Dataset initialization elapsed time: {} hours {} mins {} seconds".format(
            hour, min, sec
        )
    )
