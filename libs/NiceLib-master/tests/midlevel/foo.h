typedef struct Item Item;

extern int add(int a, int b);
extern int subtract(int a, int b);
extern Item* create_item();
extern int item_get_id(Item*);
extern float item_get_value(Item*);
extern void item_set_value(Item*, float);
extern int item_static_value(void);
