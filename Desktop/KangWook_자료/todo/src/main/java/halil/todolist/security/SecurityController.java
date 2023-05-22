package halil.todolist.security;

import halil.todolist.domain.member.repository.MemberRepository;
import halil.todolist.domain.member.service.MemberService;
import lombok.RequiredArgsConstructor;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequiredArgsConstructor
public class SecurityController {

    private final MemberService memberService;
    private final MemberRepository repository;

    @GetMapping("v1/security")
    public ResponseEntity getAll() {
        return ResponseEntity.ok(repository.findAll());
    }
}
