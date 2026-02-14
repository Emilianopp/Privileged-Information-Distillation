---
layout: default
title: Home
---

<div class="hero">
  <h1>Privileged Information Distillation</h1>
  <p class="subtitle">A research blog exploring knowledge distillation with privileged information</p>
</div>

<div class="content">
  <h2>About This Research</h2>
  <p>
    Welcome to our research blog! Here we discuss our work on privileged information distillation,
    a technique that leverages additional information available only during training to improve
    model performance at inference time.
  </p>

  <h2>Blog Posts</h2>
  <ul class="post-list">
    {% for post in site.posts %}
    <li>
      <span class="post-date">{{ post.date | date: "%B %d, %Y" }}</span>
      <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
      <p class="post-excerpt">{{ post.excerpt | strip_html | truncate: 200 }}</p>
    </li>
    {% endfor %}
  </ul>
</div>

<style>
.hero {
  text-align: center;
  padding: 2rem 0;
  margin-bottom: 2rem;
  border-bottom: 1px solid #e1e4e8;
}

.hero h1 {
  font-size: 2.5rem;
  margin-bottom: 0.5rem;
}

.subtitle {
  font-size: 1.2rem;
  color: #586069;
}

.post-list {
  list-style: none;
  padding: 0;
}

.post-list li {
  margin-bottom: 1.5rem;
  padding-bottom: 1.5rem;
  border-bottom: 1px solid #eee;
}

.post-date {
  color: #586069;
  font-size: 0.9rem;
}

.post-excerpt {
  color: #586069;
  margin-top: 0.5rem;
}
</style>
