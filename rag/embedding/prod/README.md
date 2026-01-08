
# Prod embedding pipelines

---

## 1. Markdown Docusaurus documentation

```
embedMarkdownSimpleLlamaIndexing.py
```

**Trigger**
> Gerrit merge documentation changes to dev

**Jenkins**
> https://zos-jenkins.redlabs.pl/job/dum-zos-documentation-embedding/

---

## 2. PDF specifications

```
embedPdfLlamaIndexing.py
```

**Trigger**
> Automatically everyday on midnight

**Jenkins**
> https://zos-jenkins.redlabs.pl/job/dum-zos-pdf-embedding/

---

## 3. Confluence specifications

```
embedConfluenceLlamaIndexing.py
```

**Trigger**
> Automatically everyday on midnight

**Jenkins**
> https://zos-jenkins.redlabs.pl/job/dum-zos-confluence-embedding/