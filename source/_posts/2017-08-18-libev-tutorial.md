---
title: liev tutorial
---
### 常见接口
libev通过提供几个简单的接口，可以方便设置对fd的read/write等事件进行回调。定义的loop用来循环监控事件，loop可以监听的watcher都通过
ev_TYPE定义的watcher(TYPE可以为io, signal, async, timer等事件)
>
	// define a new io watcher
	ev_io io_watcher;
	
	// bind the watcher with a io callback func
	// 将回调函数io_callback_func绑定到io_wathcer
	// 当有事件时候，回调这个函数
	ev_init(&io_watcher, io_callback_func);
	
	// watcher到具体的某个fd的READ事件
	ev_io_set(&io_watcher, fd, EV_READ);
	
	// 将watcher绑定到全局的loop上
	ev_io_start(loop, &io_watcher);
	
	// 将loop run起来。
	ev_run(loop, 0);
	
上述的ev_init以及ev_io_set可以通过一个函数来实现
> 
	// 这里的callbackfn，fd， READ事件都是用来初始化io_watcher
	// 用一个init函数更合理。
	ev_io_init(&io_watcher, io_callback_func, fd, EV_READ);

	