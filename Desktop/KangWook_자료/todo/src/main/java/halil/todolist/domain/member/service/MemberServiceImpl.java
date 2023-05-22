package halil.todolist.domain.member.service;

import halil.todolist.domain.member.dto.SignUpDto;
import halil.todolist.domain.member.entity.Member;
import halil.todolist.domain.member.exception.session.EmailDuplicate;
import halil.todolist.domain.member.exception.session.LoginUserNotFound;
import halil.todolist.domain.member.repository.MemberRepository;
import halil.todolist.security.filter.AuthenticationFilter;
import halil.todolist.security.manager.CustomAuthenticationManager;
import lombok.RequiredArgsConstructor;
import org.springframework.security.authentication.BadCredentialsException;
import org.springframework.security.authentication.UsernamePasswordAuthenticationToken;
import org.springframework.security.core.Authentication;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import javax.servlet.http.HttpServletResponse;
import java.util.Optional;

@Service
@RequiredArgsConstructor
public class MemberServiceImpl implements MemberService {

    private final MemberRepository memberRepository;
    private final PasswordEncoder passwordEncoder;
    private final CustomAuthenticationManager customAuthenticationManager;

    @Override
    @Transactional
    public Long signUp(SignUpDto signUpDto) {
        checkEmailDuplicate(signUpDto.getEmail());
        Long id = memberRepository.save(
                        Member.builder()
                                .email(signUpDto.getEmail())
                                .password(passwordEncoder.encode(signUpDto.getPassword()))
                                .build())
                .getId();

        return id;
    }

    @Override
    @Transactional
    public String login(String email, String password) {
        Optional<Member> member = memberRepository.findByEmail(email);
        if (member.isEmpty()) {
            return null;
        }

        if (passwordEncoder.matches(password, member.get().getPassword())) {
            return member.get().getEmail();
        } else {
            return null;
        }
    }

    @Override
    public void checkEmailDuplicate(String email) {
        if (memberRepository.findByEmail(email).isPresent()) {
            throw new EmailDuplicate();
        }
    }
}
