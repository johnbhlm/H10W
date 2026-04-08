std_msgs/Header header
uint8 status # 0: not initialized, 1: ready, 2: running
uint8 left_state    #左手状态 0-未持有物品，1-持有物品
uint8 right_state    #右手状态 0-未持有物品，1-持有物品
string left_item    #左手持有物品名称
string right_item    #右手持有物品名称