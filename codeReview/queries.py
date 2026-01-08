CPP_DECLS_CALLS_QUERY = r"""
; =======================
; CALLS (std filtered)
; =======================

; member calls: obj.m(...), ptr->m(...)
(call_expression
  function: (field_expression
    field: (field_identifier) @call.method
  )
)
(#match? @call.method "^(?!std::)")

; free / qualified calls: foo(...), ns::foo(...)
(call_expression
  function: [
    (identifier) @call.name
    (qualified_identifier) @call.name
  ]
)
; filter out std::... (qualified) and plain std (safety)
(#match? @call.name "^(?!std::)")

; NOTE: some grammars may tokenize `std::move` as just `identifier: move`.
; We'll *optionally* drop those in a tiny post-filter (see below) when the
; token is preceded by 'std::' in source text.

; constructor calls: new Type(...)
(new_expression
  type: [
    (type_identifier) @call.ctor
    (qualified_identifier) @call.ctor
  ]
)
(#match? @call.ctor "^(?!std::)")

; =======================
; DECLARATIONS (std filtered)
; =======================

; function definitions with body
(function_definition
  declarator: (function_declarator
    declarator: [
      (identifier) @decl.def.name
      (qualified_identifier) @decl.def.name
      (field_identifier) @decl.def.name
    ]
  )
)
(#match? @decl.def.name "^(?!std::)")

; function prototypes (no body) — via init_declarator
(declaration
  declarator: (init_declarator
    declarator: (function_declarator
      declarator: [
        (identifier) @decl.proto.name
        (qualified_identifier) @decl.proto.name
        (field_identifier) @decl.proto.name
      ]
    )
  )
)
(#match? @decl.proto.name "^(?!std::)")

; template function definitions
(template_declaration
  (function_definition
    declarator: (function_declarator
      declarator: [
        (identifier) @decl.tmpl.def.name
        (qualified_identifier) @decl.tmpl.def.name
        (field_identifier) @decl.tmpl.def.name
      ]
    )
  )
)
(#match? @decl.tmpl.def.name "^(?!std::)")

; template function prototypes
(template_declaration
  (declaration
    declarator: (init_declarator
      declarator: (function_declarator
        declarator: [
          (identifier) @decl.tmpl.name
          (qualified_identifier) @decl.tmpl.name
          (field_identifier) @decl.tmpl.name
        ]
      )
    )
  )
)
(#match? @decl.tmpl.name "^(?!std::)")

; class / struct declarations
(class_specifier
  name: (type_identifier) @decl.class.name
)
(#match? @decl.class.name "^(?!std::)")

(struct_specifier
  name: (type_identifier) @decl.struct.name
)
(#match? @decl.struct.name "^(?!std::)")
"""
