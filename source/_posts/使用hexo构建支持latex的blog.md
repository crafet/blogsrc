---
title: 使用hexo构建支持latex的blog
---

## 背景

​	之前使用了jekyll构建静态blog，但是在公式配置上一直没有配置合理，导致文章中的公式显示经常出现问题。结果反而让自己变的更懒而不勤于书写文章。偶然间看到hexo这个东东更加方便，并且起NexT主题中默认支持mathjax，不用自己再设置各种plugin来支持（事实上没有使用NexT theme之前，也设置了好久来支持公式，但是无果）。

​	因此打算将第一篇文章来描述如何使用hexo。

## 构建

​	其实随便google都能找到一堆的基于hexo搭建blog的tutorial。我这里也将自己学习的过程记录于此，方便后面查询。

* 现在安装nodejs，直接在nodejs的[官网](https://nodejs.org/zh-cn/)下载，我当前是下载了最新的[v6.11.3 LTS](https://nodejs.org/dist/v6.11.3/node-v6.11.3-x64.msi)版本，直接安装即可。

* 可以查看node的版本

  ``` bash
  node --version
  ```

* 通过rpm命令安装hexo cli

  ```bash
  npm install hexo-cli -g
  ```

* 安装完成后，可以查看安装的hexo的版本

  ```bash
  hexo -v
  ```

* 进入某个目录如test，其中初始化为blog的根路径

  ```bash
  hexo init
  ```

* 默认的hexo的主题是landscape，我们这里使用的是NexT主题，需要先clone到本地并设置_config.yml即可。

  ```bash
  git clone https://github.com/iissnan/hexo-theme-next.git themes/next
  ```

* 在_config.yml中注释掉原来的theme，并设置为theme: next

* 一般设置config后，hexo会自动reload，如果涉及到文章修改，那么一般需要以下几个命令来重新生成静态页面

  > // 清理old setting
  >
  > hexo clean 
  >
  > // g = generator,即重新生成文章
  >
  > hexo g
  >
  > // s = server,本地预览模式。一般在localhost:4000可以看到预览模式
  >
  > hexo s

* 最后是与git相关的设置，在config中设置deploy为git后，会默认提交内容到github中。因此在config中设置repo的地址即可。

  ```yaml
  deploy:
    type: git
    repo: https://github.com/crafet/crafet.github.io.git
    branch: master
  ```

  这里配置的是我的git地址，默认是master分支。配置config后，通过以下命令完成部署

  ```bash
  hexo d
  ```

  其中d=deploy。部署后将会要求输入git的账户密码进而提交到github上。**注意**这里我的repo配置的是https（公司内将ssh的22端口和443端口都禁掉了好像。会超时）。

  ​

  这样就完成了一个简单的blog构建，也可以hexo s --debug来本地进行调试(每次请求都会打印出get请求)

## 关于公式

​	其实我更看重的是公式的渲染。NexT中默认是支持mathjax的，只是默认是没打开的。在NexT自己的config中配置mathjax打开即可。

```yaml
mathjax:
  enable: true
  per_page: true
  #cdn: //cdn.bootcss.com/mathjax/2.7.1/latest.js?config=TeX-AMS-MML_HTMLorMML
  cdn: //cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-AMS-MML_HTMLorMML
```

**注意**这里的cdn，被注释掉的一行是NexT默认的cdn地址，在调试的时候发现公式是不支持的。后来google了才知道换成了新的cdn。换成新的cdn后，再次clean以及depoy就可以看到公式已经正常支持了。

​	另外基于我已经有的设置发现对于下标并不能完美支持，如公式$ J_\alpha(x) $，一般的markdown写法是

```yaml
J_\alpha
```

但是发现并不能渲染成功，google发现是因为这里的下划"_"需要转义才行即需要写成

```yaml
J\_\alpha
```

才能最终成功的渲染成功。

## 结束

啰嗦这么多成为用hexo写的第一个blog，在使用hexo以及NexT的过程中会发现其他一些问题，边学边用。